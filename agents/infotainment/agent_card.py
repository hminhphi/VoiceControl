from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from prompts import build_orchestrator_description

TELL_JOKE = AgentSkill(
    id="tell_joke",
    name="Tell joke",
    description="Tell a joke; returns path to joke file.",
    tags=["joke", "funny", "humor"],
    examples=["tell me a joke", "something funny", "a joke please"],
)
PLAY_MUSIC = AgentSkill(
    id="play_music",
    name="Play music",
    description="Play a song; returns path to audio file.",
    tags=["music", "song", "play", "audio"],
    examples=["play music", "play a song", "play Morning Drive"],
)


def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Infotainment Agent",
        description=build_orchestrator_description(),
        url=base_url.rstrip("/") + "/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(),
        skills=[TELL_JOKE, PLAY_MUSIC],
    )
