"""
Car Control UI — Web dashboard mô phỏng xe hơi
Port: 8010

Giao diện:
  - SVG car visualization với door/trunk/light/AC/window state
  - Buttons gửi lệnh đến car_control agent trực tiếp (POST /control)
  - SSE subscribe để cập nhật state realtime
  - Giao tiếp với orchestrator qua text input (demo mode)
"""
import os
import sys
import json
import asyncio
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse, JSONResponse
from typing import AsyncIterator

# ── Config ───────────────────────────────────────────────────────────────────
CAR_CONTROL_URL  = os.environ.get("CAR_CONTROL_URL",  "http://localhost:8001")
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
UI_PORT          = int(os.environ.get("CAR_CONTROL_UI_PORT", "8010"))

# URL orchestrator cho BROWSER (WebSocket + health). Trong Docker,
# ORCHESTRATOR_URL la hostname noi bo ("orchestrator") — browser khong resolve
# duoc nen WebSocket chet va khong hien cau tra loi. Mac dinh suy ra tu host
# cua request (localhost hay IP LAN deu dung); co the ghi de bang env.
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", "8000"))
ORCHESTRATOR_PUBLIC_URL = os.environ.get("ORCHESTRATOR_PUBLIC_URL", "").strip()

# File .env de panel voice lab doc/ghi (mount tu repo).
ENV_FILE = os.environ.get("ENV_FILE", "/app/.env")

# Cac khoa tien xu ly am thanh hien trong panel Voice Lab.
VOICE_KEYS = [
    "AEC_NS_LEVEL",
    "AEC_NS_LINEAR",
    "AEC_HPF_FULL_BAND",
    "AEC_TRANSIENT_SUPPRESS",
    "AEC_AGC_ENABLED",
    "AEC_LIMITER_ENABLED",
    "AEC_LIMITER_CEILING_DBFS",
    "AEC_LIMITER_KNEE",
    "AEC_AGC1_ENABLED",
    "AEC_AGC1_TARGET_DBFS",
    "AEC_DELAY_MS",
]

_HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = _HERE / "templates"

app = FastAPI(title="Car Control UI")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Browser-facing orchestrator URL: derive from the request host so it works
    # from any machine (localhost, LAN IP). The internal hostname ("orchestrator")
    # is Docker-only and breaks the WebSocket in the browser.
    public_orch = ORCHESTRATOR_PUBLIC_URL or (
        f"{request.url.scheme}://{request.url.hostname}:{ORCHESTRATOR_PORT}"
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "car_control_url": CAR_CONTROL_URL,
            "orchestrator_url": public_orch,
        },
    )


@app.get("/api/state")
async def proxy_state():
    """Proxy /state from car_control."""
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            r = await c.get(f"{CAR_CONTROL_URL}/state")
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/api/control")
async def proxy_control(request: Request):
    """Proxy POST /control to car_control."""
    body = await request.json()
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            r = await c.post(f"{CAR_CONTROL_URL}/control", json=body)
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/api/message")
async def proxy_message(request: Request):
    """Send text message to orchestrator."""
    body = await request.json()
    async with httpx.AsyncClient(timeout=60) as c:
        try:
            r = await c.post(
                f"{ORCHESTRATOR_URL}/v1/orchestrator/message",
                json=body,
            )
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)


def _read_env_keys():
    """Doc cac khoa voice trong .env (giu nguyen dong, khong parse YAML)."""
    out = {}
    try:
        with open(ENV_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                if k in VOICE_KEYS:
                    out[k] = v.strip()
    except FileNotFoundError:
        return {"error": f"ENV_FILE not found: {ENV_FILE}"}
    except Exception as e:
        return {"error": str(e)}
    return out


@app.get("/api/voice_config")
async def voice_config_get():
    return JSONResponse(_read_env_keys())


@app.post("/api/voice_config")
async def voice_config_set(request: Request):
    """Cap nhat cac khoa voice trong .env, giu nguyen phan con lai.

    Ap dung cho lan chay voice_processing KE TIEP (can restart service).
    """
    body = await request.json()
    updates = {k: str(v) for k, v in (body or {}).items() if k in VOICE_KEYS}
    if not updates:
        return JSONResponse({"error": "no valid keys", "allowed": VOICE_KEYS},
                            status_code=400)
    try:
        with open(ENV_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return JSONResponse({"error": f"ENV_FILE not found: {ENV_FILE}"},
                            status_code=503)

    seen = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k = s.split("=", 1)[0].strip()
        if k in updates:
            lines[i] = f"{k}={updates[k]}\n"
            seen.add(k)

    # Khoa chua co trong file -> them vao cuoi.
    missing = [k for k in updates if k not in seen]
    if missing:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        for k in missing:
            lines.append(f"{k}={updates[k]}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return JSONResponse({"saved": updates, "env_file": ENV_FILE,
                         "note": "restart voice_processing de ap dung"})


@app.get("/api/events")
async def proxy_events():
    """Proxy SSE /events from car_control."""
    async def _gen() -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream("GET", f"{CAR_CONTROL_URL}/events") as r:
                    async for line in r.aiter_lines():
                        if line:
                            yield line + "\n"
                        else:
                            yield "\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    print(f"[car_control_ui] Starting on http://localhost:{UI_PORT}")
    print(f"[car_control_ui] Car Control: {CAR_CONTROL_URL}")
    print(f"[car_control_ui] Orchestrator: {ORCHESTRATOR_URL}")
    uvicorn.run(app, host="0.0.0.0", port=UI_PORT)
