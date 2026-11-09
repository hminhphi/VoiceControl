import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from agent_card import build_agent_card
from agent_executor import CarControlAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
import uvicorn

PORT = int(os.environ.get("CAR_CONTROL_PORT", "8001"))
BASE_URL = os.environ.get("CAR_CONTROL_BASE_URL", f"http://localhost:{PORT}")
SOUNDS_DIR = os.environ.get("CAR_CONTROL_SOUNDS_DIR", "/app/sounds")


def _find_sound_by_segments(sounds_dir: str, segments: list[str]) -> str | None:
    if not segments or not os.path.isdir(sounds_dir):
        return None
    segments = [s.strip().lower() for s in segments if s.strip()]
    if not segments:
        return None
    for name in os.listdir(sounds_dir):
        if not name.endswith((".wav", ".mp3", ".ogg")):
            continue
        name_lower = name.lower()
        if all(seg in name_lower for seg in segments):
            return name
    return None


async def sounds_resolve(request):
    segments_str = request.query_params.get("segments", "")
    segments = [s.strip() for s in segments_str.split(",") if s.strip()]
    filename = _find_sound_by_segments(SOUNDS_DIR, segments)
    if not filename:
        return JSONResponse({"url": None}, status_code=404)
    base = BASE_URL.rstrip("/")
    url = f"{base}/static/sounds/{filename}"
    return JSONResponse({"url": url})


agent_card = build_agent_card(BASE_URL)
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

# State API (same as main_pc.py): lets the orchestrator see live car state in
# the LLM context and the UI mirror the simulator on Jetson (/state, /control).
if os.environ.get("CAR_CONTROL_ENABLE_STATE_API", "1").lower() in ("1", "true", "yes"):
    try:
        from state_server import mount as mount_state

        mount_state(app)
        print("[car_control] state_server mounted: /state /control /events", flush=True)
    except Exception as e:  # pragma: no cover - startup resilience
        print(f"[car_control][main] state_server mount failed: {e}", flush=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
