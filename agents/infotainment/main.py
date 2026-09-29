import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from agent_card import build_agent_card
from agent_executor import InfotainmentAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse
import uvicorn

PORT = int(os.environ.get("INFOTAINMENT_PORT", "8004"))
BASE_URL = os.environ.get("INFOTAINMENT_BASE_URL", f"http://localhost:{PORT}")
MEDIA_DIR = Path(os.environ.get("INFOTAINMENT_MEDIA_DIR", str(Path(__file__).resolve().parent / "media")))
JOKES_DIR = MEDIA_DIR / "jokes"
SONGS_DIR = MEDIA_DIR / "songs"

agent_card = build_agent_card(BASE_URL)
request_handler = DefaultRequestHandler(
    agent_executor=InfotainmentAgentExecutor(base_url=BASE_URL),
    task_store=InMemoryTaskStore(),
)
app_builder = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
app = app_builder.build()

@app.route("/health", methods=["GET"])
async def health(request):
    return JSONResponse({"status": "ok"})

if JOKES_DIR.is_dir():
    app.mount("/static/jokes", StaticFiles(directory=str(JOKES_DIR)), name="jokes")
if SONGS_DIR.is_dir():
    app.mount("/static/songs", StaticFiles(directory=str(SONGS_DIR)), name="songs")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
