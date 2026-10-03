import json
import os

import requests

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


def _parse_ws_url(base_url):
    base = (base_url or "").strip().rstrip("/")
    if base.startswith("https://"):
        return base.replace("https://", "wss://", 1)
    if base.startswith("http://"):
        return base.replace("http://", "ws://", 1)
    return "ws://" + base if base else ""


def send_and_stream(
    user_message,
    session_id,
    on_segment,
    base_url=None,
    n_chunks=5,
    post_timeout=30,
    ws_timeout=60,
    language=None,
    abort_event=None,
):
    """
    POST user_message to orchestrator, then consume WebSocket stream.
    Flush to TTS on sentence end (. ! ?) or on done/error.
    `language` is the STT-detected language hint so the LLM replies in-language.
    `abort_event` (threading.Event): when set (barge-in / wake-word interrupt)
    the WS stream is closed immediately and pending tokens are dropped.
    """
    base_url = base_url or os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
    base_url = base_url.rstrip("/")

    post_url = f"{base_url}/v1/orchestrator/message"
    ws_url = _parse_ws_url(base_url) + f"/v1/orchestrator/ws/{session_id}"

    payload = {"message": user_message, "session_id": session_id}
    if language:
        payload["language"] = language

    try:
        r = requests.post(
            post_url,
            json=payload,
            timeout=post_timeout,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        try:
            accepted = r.json()
            print(
                "[ORCH][router] "
                f"agent_ids={accepted.get('agent_ids')} "
                f"scores={accepted.get('agent_scores')} "
                f"threshold={accepted.get('route_threshold')}"
            )
        except Exception:
            pass
    except Exception as e:
        print(f"[orchestrator_client] POST failed: {e}")
        return

    if not HAS_WEBSOCKET:
        print("[orchestrator_client] websocket-client not installed, skip stream")
        return

    buffer = []
    SENTENCE_END = (".", "!", "?", "。", "！", "？", "\n")

    def flush():
        nonlocal buffer
        if not buffer:
            return
        segment = "".join(buffer).strip()
        buffer.clear()
        if segment:
            try:
                on_segment(segment)
            except Exception as e:
                print(f"[orchestrator_client] on_segment error: {e}")

    def on_ws_message(ws, raw):
        if abort_event is not None and abort_event.is_set():
            # Superseded turn: drop the payload and close the stream now so a
            # queued follow-up turn can start immediately.
            try:
                ws.close()
            except Exception:
                pass
            return
        try:
            d = json.loads(raw)
        except Exception:
            return
        t = d.get("type", "")
        if t == "token":
            content = d.get("content", "")
            if content:
                buffer.append(content)
                text = "".join(buffer).rstrip()
                # Flush immediately on sentence boundary or if chunk gets long at clause boundary
                if text.endswith(SENTENCE_END) or (len(text) >= 100 and text.endswith((",", "，", ";", "；"))):
                    flush()
        elif t == "session_log":
            content = d.get("content", "")
            if content:
                print(f"[ORCH][log] {content}")
        elif t in ("done", "error"):
            flush()
            try:
                ws.close()
            except Exception:
                pass

    def on_error(ws, err):
        print(f"[orchestrator_client] WebSocket error: {err}")

    def on_close(ws, close_status, close_msg):
        flush()

    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_ws_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever(ping_interval=20, ping_timeout=10)
