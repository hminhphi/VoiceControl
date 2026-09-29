import json
import logging
import os
import re
from pathlib import Path
 
from typing_extensions import override
 
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
 
from query_rewriter import rewrite_query_with_llm
from web_search import WebSearchClient
 
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("a2a").setLevel(logging.WARNING)
logger = logging.getLogger("cloud")

DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_FACTS_PATH = DATA_DIR / "demo_facts.json"
LOCAL_FACT_THRESHOLD = 0.60
STOPWORDS = {
    "a",
    "an",
    "and",
    "is",
    "of",
    "the",
    "who",
}

FACT_ROLE_QUERY_TOKENS = {
    "ceo": {"ceo"},
    "cfo": {"cfo"},
    "cio": {"cio", "io"},
}

EXECUTIVE_QUERY_TOKENS = {
    "cao",
    "ceo",
    "cfo",
    "cio",
    "cto",
    "chief",
    "company",
    "corporation",
    "executive",
    "officer",
}

COMPANY_HINT_TOKENS = {
    "denso",
    "honda",
    "mitsubishi",
    "mitsu",
    "motor",
    "motors",
    "mr",
    "nissan",
    "obc",
    "obisse",
    "toyota",
}

LLM_REWRITE_MIN_CONFIDENCE = 0.50
 
# Keywords that indicate a news-type query
NEWS_KEYWORDS = [
    "news", "latest", "breaking", "update", "headline",
]

ACRONYM_REWRITES = [
    (re.compile(r"\bC\s*\.\s*I\s*\.\s*O\.?", re.IGNORECASE), "CIO"),
    (re.compile(r"\bC\s*\.\s*F\s*\.\s*O\.?", re.IGNORECASE), "CFO"),
    (re.compile(r"\bC\s*\.\s*E\s*\.\s*O\.?", re.IGNORECASE), "CEO"),
]

MITSUBISHI_MISHEAR_REWRITES = [
    re.compile(r"\bmitsu\b", re.IGNORECASE),
    re.compile(r"\bmr\s*\.?\s*obc\b", re.IGNORECASE),
    re.compile(r"\bmr\s*\.?\s*b\s*\.?\s*c\s*\.?\b", re.IGNORECASE),
    re.compile(r"\bmr\s*\.?\s*bichie\b", re.IGNORECASE),
    re.compile(r"\bmiss?\s+obc\b", re.IGNORECASE),
    re.compile(r"\bmr\s*\.?\s*wissi\b", re.IGNORECASE),
    re.compile(r"\bmr\s*\.?\s*obisse\b", re.IGNORECASE),
    re.compile(r"\bmiss?\s+subishi\b", re.IGNORECASE),
]
 
 
def _get_user_text(context: RequestContext) -> str:
    """Extract plain text from an A2A RequestContext."""
    msg = getattr(context, "message", None)
    if not msg:
        request = getattr(context, "request", None)
        if request:
            params = getattr(request, "params", None)
            if params:
                msg = getattr(params, "message", None)
    if not msg:
        return ""
    parts = getattr(msg, "parts", None) or []
    for p in parts:
        part_root = getattr(p, "root", p)
        kind = getattr(part_root, "kind", None) or getattr(p, "type", None)
        if kind == "text":
            return getattr(part_root, "text", None) or getattr(p, "text", None) or ""
    return ""
 
 
def _is_news_query(text: str) -> bool:
    """Return True if the query looks like a news search."""
    lower = text.lower()
    return any(kw in lower for kw in NEWS_KEYWORDS)


def _rewrite_demo_fact_query(text: str) -> str:
    rewritten = text or ""
    for pattern, replacement in ACRONYM_REWRITES:
        rewritten = pattern.sub(replacement, rewritten)
    for pattern in MITSUBISHI_MISHEAR_REWRITES:
        rewritten = pattern.sub("Mitsubishi", rewritten)
    return " ".join(rewritten.split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _content_tokens(text: str) -> set[str]:
    return {tok for tok in _tokens(text) if tok not in STOPWORDS}


def _looks_like_executive_company_query(text: str) -> bool:
    tokens = _content_tokens(text)
    if not tokens & EXECUTIVE_QUERY_TOKENS:
        return False
    if tokens & COMPANY_HINT_TOKENS:
        return True
    normalized = _normalize(text)
    return bool(re.search(r"\bof\s+\w+", normalized, re.IGNORECASE))


def _required_role_tokens(fact: dict) -> set[str]:
    tags = {str(tag).lower() for tag in fact.get("tags") or []}
    if "executive" not in tags:
        return set()
    for role, query_tokens in FACT_ROLE_QUERY_TOKENS.items():
        if role in tags:
            return query_tokens
    return set()


def _load_demo_facts(path: Path = DEMO_FACTS_PATH) -> list[dict]:
    if not path.is_file():
        logger.info("[cloud.local_facts] no data file at %s", path)
        return []
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.exception("[cloud.local_facts] failed to load %s: %s", path, exc)
        return []

    facts = []
    if not isinstance(raw, list):
        logger.warning("[cloud.local_facts] expected list in %s", path)
        return facts

    for item in raw:
        if not isinstance(item, dict):
            continue
        q = (item.get("q") or "").strip()
        a = (item.get("a") or "").strip()
        if not q or not a:
            continue
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        facts.append({
            "q": q,
            "a": a,
            "aliases": [str(x).strip() for x in aliases if str(x).strip()],
            "tags": [str(x).strip() for x in tags if str(x).strip()],
        })
    logger.info("[cloud.local_facts] loaded %d facts from %s", len(facts), path)
    return facts


def _score_demo_fact(query: str, fact: dict) -> float:
    query_tokens = _content_tokens(query)
    required_role_tokens = _required_role_tokens(fact)
    if required_role_tokens and not (query_tokens & required_role_tokens):
        return 0.0

    query_norm = _normalize(query)
    candidates = [fact.get("q", ""), *(fact.get("aliases") or [])]
    for candidate in candidates:
        if query_norm and query_norm == _normalize(candidate):
            return 1.0

    best = 0.0
    for candidate in candidates:
        candidate_tokens = _content_tokens(candidate)
        if not query_tokens or not candidate_tokens:
            continue
        overlap = query_tokens & candidate_tokens
        score = len(overlap) / max(len(query_tokens), len(candidate_tokens))
        best = max(best, score)
    return best
 
 
class CloudAgentExecutor(AgentExecutor):
    """Cloud agent that uses Tavily Search API for news and general web queries."""
 
    def __init__(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        self._client = WebSearchClient(api_key)
        self._demo_facts = _load_demo_facts()
        logger.info("CloudAgentExecutor initialised (Tavily key=%s...)", api_key[:8] if api_key else "MISSING")

    def _find_demo_fact(self, query: str) -> tuple[dict | None, float]:
        best_fact = None
        best_score = 0.0
        for fact in self._demo_facts:
            score = _score_demo_fact(query, fact)
            if score > best_score:
                best_fact = fact
                best_score = score
        if best_fact is not None and best_score >= LOCAL_FACT_THRESHOLD:
            return best_fact, best_score
        return None, best_score

    async def _maybe_llm_rewrite_query(self, query: str, fact_score: float) -> tuple[str, dict]:
        if fact_score >= LOCAL_FACT_THRESHOLD:
            return query, {"attempted": False, "reason": "local_fact_matched"}
        if not _looks_like_executive_company_query(query):
            return query, {"attempted": False, "reason": "not_executive_company_query"}

        result = await rewrite_query_with_llm(query)
        rewritten_query = result.get("query") or query
        confidence = float(result.get("confidence") or 0.0)
        changed = bool(result.get("changed")) and rewritten_query != query
        accepted = changed and confidence >= LLM_REWRITE_MIN_CONFIDENCE
        diagnostics = {
            "attempted": True,
            "changed": changed,
            "accepted": accepted,
            "confidence": round(confidence, 4),
            "query": rewritten_query,
            "model": result.get("model"),
        }
        if result.get("error"):
            diagnostics["error"] = result.get("error")
        if result.get("skipped"):
            diagnostics["skipped"] = result.get("skipped")

        logger.info(
            "[cloud.query_rewrite] attempted query=%r rewritten=%r changed=%s accepted=%s confidence=%.3f model=%s",
            query,
            rewritten_query,
            changed,
            accepted,
            confidence,
            result.get("model"),
        )
        if not accepted:
            return query, diagnostics
        return rewritten_query, diagnostics
 
    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
 
        async def _out(success: bool, message: str, image=None, documents=None, **meta) -> None:
            body = {
                "success": success,
                "message": message,
                "data": {
                    "image": image,
                    "documents": documents or []
                },
                "meta": meta or {}
            }
 
            await event_queue.enqueue_event(
                new_agent_text_message(json.dumps(body))
            )
 
        text = _get_user_text(context)
        if not text.strip():
            await _out(
                False,
                "Please send a search query, e.g. 'latest AI news' or 'Tesla company info'."
            )
            return
 
        original_query = text.strip()
        query = _rewrite_demo_fact_query(original_query)
        logger.info("Query: %s", original_query)
        if query != original_query:
            logger.info(
                "[cloud.local_facts] rewritten query=%r -> %r",
                original_query,
                query,
            )

        fact, fact_score = self._find_demo_fact(query)
        llm_rewrite = {"attempted": False}
        if fact is None:
            llm_query, llm_rewrite = await self._maybe_llm_rewrite_query(query, fact_score)
            if llm_query != query:
                llm_fact, llm_fact_score = self._find_demo_fact(llm_query)
                if llm_fact is not None:
                    logger.info(
                        "[cloud.local_facts] matched after LLM rewrite query=%r score=%.3f q=%r",
                        llm_query,
                        llm_fact_score,
                        llm_fact.get("q", ""),
                    )
                    query = llm_query
                    fact = llm_fact
                    fact_score = llm_fact_score
                else:
                    logger.info(
                        "[cloud.local_facts] no_match after LLM rewrite query=%r best_score=%.3f",
                        llm_query,
                        llm_fact_score,
                    )
                    query = llm_query
                    fact_score = llm_fact_score

        if fact is not None:
            logger.info(
                "[cloud.local_facts] matched query=%r score=%.3f q=%r",
                query,
                fact_score,
                fact.get("q", ""),
            )
            documents = [{
                "q": fact.get("q", ""),
                "a": fact.get("a", ""),
                "url": "",
                "source": "local_demo_facts",
                "tags": fact.get("tags", []),
            }]
            await _out(
                True,
                message=fact.get("a", ""),
                documents=documents,
                query=query,
                original_query=original_query,
                rewritten_query=query,
                llm_rewrite_attempted=llm_rewrite.get("attempted", False),
                llm_rewrite_changed=llm_rewrite.get("changed", False),
                llm_rewrite_accepted=llm_rewrite.get("accepted", False),
                llm_rewrite_confidence=llm_rewrite.get("confidence"),
                llm_rewrite_query=llm_rewrite.get("query"),
                llm_rewrite_model=llm_rewrite.get("model"),
                source="local_demo_facts",
                score=round(float(fact_score), 4),
                matched_q=fact.get("q", ""),
            )
            return

        logger.info(
            "[cloud.local_facts] no_match query=%r best_score=%.3f threshold=%.3f; falling back to Tavily",
            query,
            fact_score,
            LOCAL_FACT_THRESHOLD,
        )
 
        # Route to news or general search based on keywords
        if _is_news_query(query):
            logger.info("Detected NEWS query")
            result = self._client.search_news(query)
        else:
            logger.info("Detected GENERAL query")
            result = self._client.search_general(query)
 
        # Check for errors
        if result.get("error"):
            await _out(
                False,
                f"Search failed: {result['error']}",
                query=query,
                original_query=original_query,
                rewritten_query=query,
                source="web_search",
                local_fact_best_score=round(float(fact_score), 4),
                local_fact_threshold=round(float(LOCAL_FACT_THRESHOLD), 4),
                llm_rewrite_attempted=llm_rewrite.get("attempted", False),
                llm_rewrite_changed=llm_rewrite.get("changed", False),
                llm_rewrite_accepted=llm_rewrite.get("accepted", False),
                llm_rewrite_confidence=llm_rewrite.get("confidence"),
                llm_rewrite_query=llm_rewrite.get("query"),
                llm_rewrite_model=llm_rewrite.get("model"),
                llm_rewrite_error=llm_rewrite.get("error"),
                llm_rewrite_skipped=llm_rewrite.get("skipped"),
            )
            return
 
        sources = result.get("sources") or []
        images = result.get("images") or []
 
        # Convert sources to "documents" format similar to car_manual
        documents = []
        for s in sources:
            # Clean snippet: remove markdown characters that clutter the output (#, *)
            clean_a = s.get("snippet", "").replace("#", "").replace("*", "").strip()
            documents.append({
                "q": s.get("title", ""),
                "a": clean_a,
                "url": s.get("url", ""),
                "source": "web_search"
            })
 
        # Build message from document snippets (no answer mode)
        if documents:
            message_text = "\n\n".join([doc["a"] for doc in documents])
        else:
            message_text = "No results found for your query."
 
        first_image = images[0] if images else ""
 
        await _out(
            True,
            message=message_text,
            image=first_image,
            documents=documents,
            query=query,
            original_query=original_query,
            rewritten_query=query,
            source="web_search",
            local_fact_best_score=round(float(fact_score), 4),
            local_fact_threshold=round(float(LOCAL_FACT_THRESHOLD), 4),
            llm_rewrite_attempted=llm_rewrite.get("attempted", False),
            llm_rewrite_changed=llm_rewrite.get("changed", False),
            llm_rewrite_accepted=llm_rewrite.get("accepted", False),
            llm_rewrite_confidence=llm_rewrite.get("confidence"),
            llm_rewrite_query=llm_rewrite.get("query"),
            llm_rewrite_model=llm_rewrite.get("model"),
            llm_rewrite_error=llm_rewrite.get("error"),
            llm_rewrite_skipped=llm_rewrite.get("skipped"),
        )
 
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")
 
 
# ── Standalone CLI test ─────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    from dotenv import load_dotenv
 
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
 
    class MockEventQueue:
        async def enqueue_event(self, event):
            try:
                text = event.parts[0].text
            except AttributeError:
                try:
                    text = event.parts[0].root.text
                except AttributeError:
                    text = str(event)
            print(f"OUT: {text}")
 
    async def main():
        executor = CloudAgentExecutor()
        queue = MockEventQueue()
 
        print("\nCloud Agent CLI (Tavily Web Search)")
        print("Type a query, or :q to quit.\n")
 
        while True:
            try:
                user_input = input("Cloud > ").strip()
                if user_input.lower() in [":q", "exit", "quit"]:
                    break
                if not user_input:
                    continue
 
                class _Part:
                    def __init__(self, text):
                        self.root = type("obj", (object,), {"kind": "text", "text": text})()
 
                class _Msg:
                    def __init__(self, text):
                        self.parts = [_Part(text)]
 
                class _Params:
                    def __init__(self, text):
                        self.message = _Msg(text)
 
                class MockRequest:
                    def __init__(self, text):
                        self.request = type("obj", (object,), {"params": _Params(text)})()
 
                context = MockRequest(user_input)
                await executor.execute(context, queue)
            except KeyboardInterrupt:
                break
 
    asyncio.run(main())
 
