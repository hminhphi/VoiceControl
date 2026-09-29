import logging
from tavily import TavilyClient
 
from utils import read_config
 
logger = logging.getLogger("cloud.web_search")
 
 
class WebSearchClient:
    """Thin wrapper around Tavily Search API."""
 
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required.")
        self._client = TavilyClient(api_key=api_key)
       
        self.config = read_config("config/config.yaml").get("web_search", {})
        self.max_results = self.config.get("max_result", 3)
        self.include_answer = self.config.get("include_answer", False)
        self.include_images = self.config.get("include_images", True)
        self.search_depth = self.config.get("search_depth", "ultra-fast")
 
    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
 
    def search_news(self, query: str) -> dict:
        """Search latest news (topic='news')."""
        return self._search(query, topic="news")
 
    def search_general(self, query: str) -> dict:
        """General web search (topic='general')."""
        return self._search(query, topic="general")
 
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
 
    def _search(self, query: str, topic: str = "general") -> dict:
        """Call Tavily and return a formatted dict."""
        try:
            response = self._client.search(
                query=query,
                topic=topic,
                search_depth=self.search_depth,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_images=self.include_images,
            )
           
            return self._format(response)
        except Exception as exc:
            logger.exception("Tavily search failed: %s", exc)
            return {"sources": [], "images": [], "error": str(exc)}
 
    @staticmethod
    def _format(response: dict) -> dict:
        """Normalise Tavily response to a clean dict (sources + images only)."""
        sources = []
        for r in response.get("results") or []:
            sources.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:300],
            })
 
        images = response.get("images", [])
        if not isinstance(images, list):
            images = []
        return {
            "sources": sources,
            "images": images,
        }
 
 
# ── standalone test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
 
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
 
    api_key = os.environ.get("TAVILY_API_KEY", "")
    client = WebSearchClient(api_key)
 
    print("=== News search ===")
    result = client.search_news("The news of Tesla")
    print(f"result: {result}")
 
    print("\n=== General search ===")
    result = client.search_general("VinFast company info")
    print(f"result: {result}")
 
 