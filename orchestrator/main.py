import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
 
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
 
from dotenv import load_dotenv
 

from api.route_agents import router as agents_router
from api.route_orchestrator import router as orchestrator_router
 
 
import uvicorn
 
 
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
 
logging.basicConfig(level=logging.INFO, format="%(message)s")
 
logging.getLogger("a2a").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
 
logger = logging.getLogger("orchestrator.main")
 
PORT = int(os.environ.get("ORCHESTRATOR_PORT", "8000"))
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    yield
    logger.info("Application shutdown...")
 
 
app = FastAPI(
    title="Edge Orchestrator API",
    description="Agent orchestrator API",
    version="1.0.0",
    lifespan=lifespan,
)
 
 
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
 
    async def dispatch(self, request: Request, call_next):
 
        response: Response = await call_next(request)
 
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
 
        return response

 
@app.get("/health")
async def health_check():
    return {"status": "ok"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

app.add_middleware(SecurityHeadersMiddleware)
 
# Routers
app.include_router(agents_router, prefix="/v1")
app.include_router(orchestrator_router, prefix="/v1")
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
 