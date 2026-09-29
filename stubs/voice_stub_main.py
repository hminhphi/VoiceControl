#!/usr/bin/env python3
"""
voice_stub_main.py — PC stub thay thế voice_processing/main.py

Thay vì full audio pipeline (mic → VAD → STT → orchestrator → TTS),
stub này tạo một interactive stdin loop:
  - User gõ text → gửi đến orchestrator qua HTTP+WebSocket
  - Response in ra terminal (thay cho TTS)
  - Giao diện terminal màu sắc

Cách chạy:
  python voice_stub_main.py

Env vars được đọc:
  ORCHESTRATOR_URL  (default: http://localhost:8000)
  VOICE_STUB_SESSION_ID  (default: voice-stub-{pid})
"""
import json
import os
import sys
import time
import threading
import uuid
try:
    import readline  # noqa: F401 — enable arrow-key history on Linux/Mac
except ImportError:
    try:
        import pyreadline3 as readline  # noqa: F401
    except ImportError:
        pass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[voice_stub] WARNING: 'requests' not installed. Run: pip install requests")

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    print("[voice_stub] WARNING: 'websocket-client' not installed. Run: pip install websocket-client")

# ── ANSI colors ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"

def _c(color, text): return f"{color}{text}{RESET}"

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL    = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000").rstrip("/")
PID         = os.getpid()
SESSION_ID  = os.environ.get("VOICE_STUB_SESSION_ID", f"voice-stub-{PID}")
POST_URL    = f"{BASE_URL}/v1/orchestrator/message"
WS_BASE     = BASE_URL.replace("https://", "wss://").replace("http://", "ws://")


def _ws_url(sid): return f"{WS_BASE}/v1/orchestrator/ws/{sid}"


# ── WebSocket consumer ────────────────────────────────────────────────────────
def consume_ws(session_id: str, stop_event: threading.Event):
    """Open WS, stream tokens, print response, then set stop_event."""
    if not HAS_WEBSOCKET:
        stop_event.set()
        return

    url = _ws_url(session_id)
    full_text = []

    def on_message(ws, raw):
        try:
            d = json.loads(raw)
        except Exception:
            return
        t = d.get("type", "")
        if t == "token":
            tok = d.get("content", "")
            if tok:
                full_text.append(tok)
                print(_c(GREEN, tok), end="", flush=True)
        elif t == "session_log":
            content = d.get("content", "")
            if content:
                print(_c(DIM, f"\n  [log] {content}"), flush=True)
        elif t == "data":
            pass  # skip data events
        elif t in ("done", "error"):
            if t == "error":
                print(_c(RED, f"\n[ERROR] {d.get('message','')}"), flush=True)
            if not full_text:
                # done with no tokens — nothing to print
                pass
            print()  # newline after response
            try:
                ws.close()
            except Exception:
                pass
            stop_event.set()

    def on_error(ws, err):
        print(_c(RED, f"\n[ws error] {err}"), flush=True)
        stop_event.set()

    def on_close(ws, *a):
        stop_event.set()

    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=20, ping_timeout=10)


# ── Send to orchestrator ──────────────────────────────────────────────────────
def send_message(text: str) -> bool:
    """POST message, then consume WS stream. Return True on success."""
    if not HAS_REQUESTS:
        print(_c(RED, "[voice_stub] requests not available"))
        return False

    session_id = f"voice-stub-{uuid.uuid4().hex[:8]}"
    stop_event = threading.Event()

    # Start WS consumer thread BEFORE posting
    ws_thread = threading.Thread(
        target=consume_ws,
        args=(session_id, stop_event),
        daemon=True,
    )
    ws_thread.start()

    # Small delay to let WS connect before POST triggers processing
    time.sleep(0.15)

    try:
        r = requests.post(
            POST_URL,
            json={"message": text, "session_id": session_id},
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        accepted = r.json()
        agents  = accepted.get("agent_ids", [])
        scores  = accepted.get("agent_scores", {})
        thresh  = accepted.get("route_threshold", 0)
        score_str = ", ".join(f"{a}={scores.get(a,0):.3f}" for a in agents)
        print(_c(DIM, f"  -> routed to [{', '.join(agents)}] scores={score_str} threshold={thresh:.3f}"))
    except Exception as e:
        print(_c(RED, f"\n[voice_stub] POST failed: {e}"))
        stop_event.set()
        return False

    # Wait for WS to finish (timeout 60s)
    stop_event.wait(timeout=60)
    return True


# ── Health check ──────────────────────────────────────────────────────────────
def wait_for_orchestrator(max_retries=30, delay=2.0):
    if not HAS_REQUESTS:
        return False
    health_url = f"{BASE_URL}/health"
    for i in range(max_retries):
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        if i == 0:
            print(_c(YELLOW, f"[voice_stub] Waiting for orchestrator at {health_url} ..."))
        time.sleep(delay)
    return False


# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = f"""
{CYAN}{BOLD}========================================================
       [VOICE STUB] Voice Processing -- PC Mode
========================================================{RESET}
{DIM}Orchestrator  : {BASE_URL}
Session prefix: {SESSION_ID}
Type your message and press Enter. Type 'exit' or 'quit' to stop.{RESET}
"""


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    # Windows: enable ANSI
    if sys.platform == "win32":
        os.system("")

    print(BANNER)

    ok = wait_for_orchestrator()
    if not ok:
        print(_c(RED, "[voice_stub] Orchestrator not reachable. Starting anyway -- will retry per message."))

    print(_c(CYAN, f"\n{BOLD}Ready. Say something:{RESET}"))
    print()

    while True:
        try:
            raw = input(_c(CYAN + BOLD, "You: ") + RESET)
        except (EOFError, KeyboardInterrupt):
            print(_c(YELLOW, "\n[voice_stub] Exiting."))
            break

        text = raw.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            print(_c(YELLOW, "[voice_stub] Goodbye!"))
            break

        print(_c(MAGENTA, f"\n[Assistant]: "), end="", flush=True)
        send_message(text)
        print()


if __name__ == "__main__":
    main()
