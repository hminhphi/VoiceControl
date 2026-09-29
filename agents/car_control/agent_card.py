from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from prompts import build_orchestrator_description

DOOR_SKILL = AgentSkill(
    id="door_control",
    name="Door control",
    description="Open or close the left or right car door.",
    tags=["door", "left door", "right door", "open", "close"],
    examples=["open the left door", "close the right door"],
)

TRUNK_SKILL = AgentSkill(
    id="trunk_control",
    name="Trunk control",
    description="Open or close the trunk.",
    tags=["trunk", "open", "close"],
    examples=["open the trunk", "close the trunk"],
)

def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Car Control Agent",
        description=build_orchestrator_description(),
        url=base_url.rstrip("/") + "/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(),
        skills=[DOOR_SKILL, TRUNK_SKILL],
    )
