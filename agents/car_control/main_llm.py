import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from agent_card import build_agent_card
from agent_executor_llm import CarControlAgentExecutorLLM
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
    agent_executor=CarControlAgentExecutorLLM(base_url=BASE_URL),
    task_store=InMemoryTaskStore(),
)
app_builder = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
app = app_builder.build()

app.add_route("/sounds/resolve", sounds_resolve, ["GET"])

if os.path.isdir(SOUNDS_DIR):
    app.mount("/static/sounds", StaticFiles(directory=SOUNDS_DIR), name="sounds")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
