from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from prompts import build_orchestrator_description

NEAREST_POI = AgentSkill(
    id="nearest_poi",
    name="Nearest POI",
    description="Find nearest restaurants, shopping, eateries by type.",
    tags=["restaurant", "shopping", "eatery", "nearby", "nearest"],
    examples=["find nearby restaurant", "nearest cafe", "shopping near me", "eateries around here"],
)
DIRECTIONS = AgentSkill(
    id="directions",
    name="Directions",
    description="Get directions to a destination (address and text; no map UI).",
    tags=["directions", "route", "navigate"],
    examples=["directions to Pho 24", "how do I get to the mall", "navigate to 123 Nguyen Hue"],
)
def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Navigation Agent",
        description=build_orchestrator_description(),
        url=base_url.rstrip("/") + "/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(),
        skills=[NEAREST_POI, DIRECTIONS],
    )
