import os
from pathlib import Path

from dotenv import load_dotenv
from starlette.responses import JSONResponse

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from agent_card import build_agent_card
from agent_executor import CarManualAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
import uvicorn

PORT = int(os.environ.get("CAR_MANUAL_PORT", "8002"))
BASE_URL = os.environ.get("CAR_MANUAL_BASE_URL", f"http://localhost:{PORT}")

executor = CarManualAgentExecutor()
executor.warm_up()

agent_card = build_agent_card(BASE_URL)
request_handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),
)
app_builder = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
app = app_builder.build()

async def health(request):
    return JSONResponse({"status": "ok"})

app.add_route("/health", health, ["GET"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
