"""
state_server.py — Car Control state API extension

Khi CAR_CONTROL_ENABLE_STATE_API=1, các routes này được thêm vào car_control app:
  GET  /state          → JSON snapshot của CarSimulator
  POST /control        → apply action {component, action}
  GET  /events         → SSE stream (text/event-stream) trả state updates

Import trong main.py KHÔNG được thay đổi — được kích hoạt qua env var.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncIterator

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

# ── Shared simulator singleton ───────────────────────────────────────────────
# Use module-level singleton so agent_executor and state_server share one instance.
# car_simulator.py must expose _SHARED_SIMULATOR or we create one here.
try:
    import car_simulator as _cs_mod
    # Create singleton on first import; reuse on subsequent imports
    if not hasattr(_cs_mod, "_SHARED_SIMULATOR"):
        _cs_mod._SHARED_SIMULATOR = _cs_mod.CarSimulator()
    _simulator = _cs_mod._SHARED_SIMULATOR
except ImportError:
    _simulator = None


_LISTENERS: list[asyncio.Queue] = []

# ── Helpers ──────────────────────────────────────────────────────────────────

def _state_dict() -> dict:
    if _simulator is None:
        return {"error": "simulator not available"}
    s = _simulator.get_state()
    return {
        "door_locked":   s.door_locked,
        "door_open":     s.door_open,
        "window_open":   s.window_open,
        "trunk_open":    s.trunk_open,
        "light_on":      s.light_on,
        "ac_on":         s.ac_on,
        "mirror_folded": s.mirror_folded,
        "timestamp":     time.time(),
    }


def _broadcast_state():
    """Push state snapshot to all SSE listeners."""
    data = json.dumps(_state_dict())
    dead = []
    for q in _LISTENERS:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _LISTENERS.remove(q)
        except ValueError:
            pass


# ── Route handlers ────────────────────────────────────────────────────────────

async def get_state(request: Request) -> JSONResponse:
    return JSONResponse(_state_dict())


async def post_control(request: Request) -> JSONResponse:
    if _simulator is None:
        return JSONResponse({"success": False, "error": "simulator not available"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "invalid JSON"}, status_code=400)

    component = str(body.get("component", "")).strip().lower()
    action    = str(body.get("action", "")).strip().lower()

    dispatch = {
        "left_door":  ("set_door",   action),
        "right_door": ("set_door",   action),
        "trunk":      ("set_trunk",  action),
        "window":     ("set_window", action),
        "light":      ("set_light",  action),
        "ac":         ("set_ac",     action),
        "mirror":     ("set_mirror", action),
    }

    if component not in dispatch:
        return JSONResponse({"success": False, "error": f"unknown component: {component}"}, status_code=400)

    method_name, arg = dispatch[component]
    method = getattr(_simulator, method_name, None)
    if method is None:
        return JSONResponse({"success": False, "error": "method not found"}, status_code=500)

    ok, result = method(arg)
    _broadcast_state()

    return JSONResponse({
        "success": ok,
        "component": component,
        "action": action,
        "result": result,
        "state": _state_dict(),
    })


async def _sse_generator() -> AsyncIterator[str]:
    """SSE stream — sends full state on connect, then on each update."""
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=20)
    _LISTENERS.append(q)

    # Send initial state
    yield f"data: {json.dumps(_state_dict())}\n\n"

    try:
        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=25.0)
                yield f"data: {data}\n\n"
            except asyncio.TimeoutError:
                # heartbeat
                yield ": ping\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        try:
            _LISTENERS.remove(q)
        except ValueError:
            pass


async def get_events(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Mount helper ──────────────────────────────────────────────────────────────

def mount(app) -> None:
    """Add state routes to a Starlette/A2A app instance."""
    app.add_route("/state", get_state, methods=["GET"])
    app.add_route("/control", post_control, methods=["POST"])
    app.add_route("/events", get_events, methods=["GET"])
    print("[car_control] state_server mounted: /state /control /events", flush=True)
