from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from prompts import build_orchestrator_description

NEWS_SEARCH = AgentSkill(
    id="news_search",
    name="News Search",
    description="Search latest news, company news, trending topics via web search.",
    tags=["news", "company news", "trending", "latest", "headlines"],
    examples=[
        "latest news about AI",
        "Toyota company news",
        "trending tech news",
        "breaking news today",
    ],
)

WEB_SEARCH = AgentSkill(
    id="web_search",
    name="Web Search",
    description="Search for company information, general knowledge, or any web query.",
    tags=["search", "company", "info", "web", "lookup", "general"],
    examples=[
        "what is Tesla",
        "VinFast company info",
        "how does electric car work",
        "who is Elon Musk",
    ],
)


def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Cloud Agent",
        description=build_orchestrator_description(),
        url=base_url.rstrip("/") + "/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(),
        skills=[NEWS_SEARCH, WEB_SEARCH],
    )
