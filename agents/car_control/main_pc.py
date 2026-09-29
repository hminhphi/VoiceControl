import os
import sys
from pathlib import Path

# Ensure we're in the car_control directory
_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

# Load .env từ project root
from dotenv import load_dotenv
load_dotenv(_DIR.parent.parent / ".env")

# ── Pre-init shared simulator singleton ─────────────────────────────────────
# Both state_server and agent_executor will share this instance.
import car_simulator as _cs_mod
if not hasattr(_cs_mod, "_SHARED_SIMULATOR"):
    _cs_mod._SHARED_SIMULATOR = _cs_mod.CarSimulator()

# ── Import app từ main.py gốc ────────────────────────────────────────────────
from agent_card import build_agent_card
from agent_executor import CarControlAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
import uvicorn

PORT      = int(os.environ.get("CAR_CONTROL_PORT", "8001"))
BASE_URL  = os.environ.get("CAR_CONTROL_BASE_URL", f"http://localhost:{PORT}")
SOUNDS_DIR = os.environ.get("CAR_CONTROL_SOUNDS_DIR", str(_DIR / "sounds"))


def _find_sound_by_segments(sounds_dir, segments):
    if not segments or not os.path.isdir(sounds_dir):
        return None
    segments = [s.strip().lower() for s in segments if s.strip()]
    for name in os.listdir(sounds_dir):
        if not name.endswith((".wav", ".mp3", ".ogg")):
            continue
        if all(seg in name.lower() for seg in segments):
            return name
    return None


async def sounds_resolve(request):
    segments_str = request.query_params.get("segments", "")
    segments = [s.strip() for s in segments_str.split(",") if s.strip()]
    filename = _find_sound_by_segments(SOUNDS_DIR, segments)
    if not filename:
        return JSONResponse({"url": None}, status_code=404)
    url = f"{BASE_URL.rstrip('/')}/static/sounds/{filename}"
    return JSONResponse({"url": url})


agent_card      = build_agent_card(BASE_URL)
request_handler = DefaultRequestHandler(
    agent_executor=CarControlAgentExecutor(base_url=BASE_URL),
    task_store=InMemoryTaskStore(),
)
app_builder = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
app = app_builder.build()


async def health(request):
    return JSONResponse({"status": "ok"})

app.add_route("/health", health, ["GET"])
app.add_route("/sounds/resolve", sounds_resolve, ["GET"])

if os.path.isdir(SOUNDS_DIR):
    app.mount("/static/sounds", StaticFiles(directory=SOUNDS_DIR), name="sounds")

# ── Conditionally mount state server ─────────────────────────────────────────
print("Checking CAR_CONTROL_ENABLE_STATE_API=", os.environ.get("CAR_CONTROL_ENABLE_STATE_API", "1"), flush=True)
if os.environ.get("CAR_CONTROL_ENABLE_STATE_API", "1").lower() in ("1", "true", "yes"):
    print("Attempting to mount state server...", flush=True)
    try:
        from state_server import mount as mount_state
        mount_state(app)
    except Exception as e:
        print(f"[car_control][main_pc] state_server mount failed: {e}", flush=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
