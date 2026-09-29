import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from agent_card import build_agent_card
from agent_executor import CloudAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.responses import JSONResponse
import uvicorn

PORT = int(os.environ.get("CLOUD_PORT", "8005"))
BASE_URL = os.environ.get("CLOUD_BASE_URL", f"http://localhost:{PORT}")

executor = CloudAgentExecutor()

agent_card = build_agent_card(BASE_URL)
request_handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),
)
app_builder = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
app = app_builder.build()

@app.route("/health", methods=["GET"])
async def health(request):
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
