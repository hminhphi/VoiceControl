from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from prompts import build_orchestrator_description

MANUAL_SKILL = AgentSkill(
    id="car_manual",
    name="Car manual Q&A",
    description="Answer questions from the car manual and configured demo Q&A: how-to, settings, maintenance, automotive companies, and executives.",
    tags=[
        "manual", "how-to", "settings", "maintenance", "bluetooth", "tire", "trunk",
        "company", "executive", "cfo", "cio", "ceo", "cro", "president",
        "vice president", "vp", "chief executive officer",
        "fpt", "fpt software", "fpt automotive", "pham minh tuan",
        "tony", "cherry", "ave", "aidv",
    ],
    examples=[
        "how to open trunk",
        "where is fuel cap release",
        "reset maintenance reminder",
        "pair Bluetooth",
        "tire pressure",
        "Who is CFO of Nissan?",
        "Who is CIO of Mitsubishi?",
        "Who is President of Mitsubishi?",
        "Who is Tony?",
        "Who is Cherry?",
        "Who is CEO of FPT Automotive?",
        "Who is Vice President of FPT?",
        "Who is CEO of FPT Software?",
        "Who is Pham Minh Tuan?",
    ],
)


def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Car Manual Agent",
        description=build_orchestrator_description(),
        url=base_url.rstrip("/") + "/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(),
        skills=[MANUAL_SKILL],
    )


if __name__ == "__main__":
    import json
    card = build_agent_card("http://localhost:8002")
    print("=== Car Manual Agent Card ===\n")
    print(json.dumps(card.model_dump(), indent=2, default=str))
