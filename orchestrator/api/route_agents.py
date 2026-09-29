import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict

from api.route_orchestrator import orchestrator_router
from api.route_orchestrator import registry


logger = logging.getLogger("orchestrator.agents")

router = APIRouter(prefix="/agents", tags=["Agents"])


class AddAgentRequest(BaseModel):
    url: str
    agent_id: str


# -------------------------
# runtime agents (alive)
# -------------------------

@router.get("", response_model=List[Dict], status_code=status.HTTP_200_OK)
async def list_agents():

    try:
        return registry.get_all()

    except Exception as e:

        logger.error("Failed to list agents", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# -------------------------
# catalog (configured agents)
# -------------------------

@router.get("/catalog", response_model=Dict, status_code=status.HTTP_200_OK)
async def catalog():

    try:
        return registry.get_catalog()

    except Exception as e:

        logger.error("Failed to get catalog", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# -------------------------
# get one
# -------------------------

@router.get("/{agent_id}", response_model=Dict, status_code=status.HTTP_200_OK)
async def get_agent(agent_id: str):

    agent = registry.get(agent_id)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    return agent


# -------------------------
# add
# -------------------------

@router.post("", response_model=Dict, status_code=status.HTTP_201_CREATED)
async def add_agent(req: AddAgentRequest):

    url = (req.url or "").strip()
    agent_id = (req.agent_id or "").strip()

    if not url or not agent_id:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="url and agent_id are required",
        )

    entry = registry.add(url=url, agent_id=agent_id)

    if not entry:

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not fetch agent card from {url}",
        )
    
    orchestrator_router.reload_agents()

    return entry


# -------------------------
# remove (disable)
# -------------------------

@router.delete("/{agent_id}", status_code=status.HTTP_200_OK)
async def remove_agent(agent_id: str):

    if not registry.remove(agent_id):

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    orchestrator_router.reload_agents()

    return {"status": "removed", "agent_id": agent_id}