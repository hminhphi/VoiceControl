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
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "car_control_url": CAR_CONTROL_URL,
            "orchestrator_url": ORCHESTRATOR_URL,
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
