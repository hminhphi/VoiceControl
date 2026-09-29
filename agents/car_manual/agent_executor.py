import json
import logging
import os
import time

from typing_extensions import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from qa_store import QAStore

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("a2a").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("car_manual")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_user_text(context: RequestContext) -> str:
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


def _format_documents(result: dict) -> str:
    """Format QAStore result into a plain-text message.

    - Confident (1 doc): return the answer directly.
    - Not confident (up to 3 docs): list answers so the orchestrator
      can pick or synthesize.
    """
    documents = result.get("documents", [])
    if not documents:
        return result.get("answer", "No answer found.")

    if result.get("confident"):
        return documents[0].get("a", "")

    # Multiple documents: keep the matched question attached to each answer so
    # the orchestrator can judge relevance without another hidden classifier.
    lines = []
    for doc in documents:
        lines.append(f"Q: {doc.get('q', '')}\nA: {doc.get('a', '')}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------

class CarManualAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self._store: QAStore | None = None

    def _ensure_store(self) -> QAStore:
        if self._store is None:
            brand = os.environ.get("CAR_MANUAL_BRAND", "mmc,toyota").strip() or "mmc,toyota"
            self._store = QAStore(brand)
            self._store.load()
        return self._store

    def warm_up(self) -> None:
        """Pre-load QAStore during startup."""
        self._ensure_store()

    def _answer_query(self, store: QAStore, original_text: str) -> tuple[str, list[dict], dict]:
        result = store.answer(original_text)
        documents = [dict(doc) for doc in result.get("documents", [])]
        matched_q = documents[0].get("q") if documents else None
        score = float(result.get("score", 0.0))
        threshold = float(store.score_threshold)
        passes_threshold = score >= threshold
        message_text = _format_documents(result)
        meta = {
            "forced_demo": False,
            "original_query": original_text,
            "matched_q": matched_q,
            "score": round(score, 4),
            "local_fact_threshold": round(threshold, 4),
            "passes_threshold": passes_threshold,
            # Backward-compatible alias for callers that still read this field.
            "confident": passes_threshold,
            "num_documents": len(documents),
            "search_mode": store.search_mode,
        }
        if passes_threshold and matched_q:
            meta["forced_query"] = matched_q
            for doc in documents:
                doc["forced_query"] = matched_q
        return message_text, documents, meta

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        t0 = time.perf_counter()

        text = _get_user_text(context)
        store = self._ensure_store()

        t1 = time.perf_counter()
        message_text, documents, meta = self._answer_query(store, text)
        rag_ms = (time.perf_counter() - t1) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        meta["timing_ms"] = {
            "rag": round(rag_ms, 3),
            "total": round(total_ms, 3),
        }

        logger.info(
            "[car_manual] original_query=%r matched_q=%r brand=%s mode=%s score=%.4f threshold=%.4f passes_threshold=%s docs=%d rag_ms=%.3f total_ms=%.3f",
            text,
            meta.get("matched_q"),
            store.brand,
            store.search_mode,
            float(meta.get("score", 0.0)),
            float(meta.get("local_fact_threshold", 0.0)),
            bool(meta.get("passes_threshold", False)),
            len(documents),
            rag_ms,
            total_ms,
        )
        for rank, doc in enumerate(documents, 1):
            logger.info(
                "[car_manual] doc rank=%d source=%s score=%.4f bm25=%.4f rrf=%.6f q=%r a=%r",
                rank,
                doc.get("source"),
                float(doc.get("score", 0.0)),
                float(doc.get("bm25_score", 0.0)),
                float(doc.get("rrf_score", 0.0)),
                doc.get("q", ""),
                doc.get("a", ""),
            )

        passes_threshold = bool(meta.get("passes_threshold", False))
        response_documents = documents if passes_threshold else []
        response_message = message_text if passes_threshold else "No answer found."

        out = json.dumps({
            "success": bool(passes_threshold),
            "message": response_message,
            "data": {
                "documents": response_documents,
            },
            "meta": meta,
        })

        await event_queue.enqueue_event(new_agent_text_message(out))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


if __name__ == "__main__":
    """
    Standalone test for CarManualAgentExecutor.

    Tests the full retrieval pipeline:
      QAStore.load() -> QAStore.answer() -> score threshold -> 1 or 3 documents

    Usage:
      python agent_executor.py
      python agent_executor.py -q "How many airbags?"
    """
    import argparse
    import asyncio

    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    parser = argparse.ArgumentParser("agent_executor test")
    parser.add_argument("-q", "--query", default="How do I reset the tire pressure warning?")
    args = parser.parse_args()

    async def _test():
        executor = CarManualAgentExecutor()
        store = executor._ensure_store()

        print("\n=== Agent Executor Test ===")
        print(f"brand       : {store.brand}")
        print(f"search_mode : {store.search_mode}")
        print(f"top_k       : {store.top_k}")
        print(f"threshold   : {store.score_threshold}")
        print(f"query       : {args.query}")
        print()

        t0 = time.perf_counter()
        result = store.answer(args.query)
        rag_ms = (time.perf_counter() - t0) * 1000.0

        print("=== RAG Result ===")
        print(f"answer    : {result['answer'][:200]}")
        print(f"score     : {result['score']:.4f}")
        print(f"confident : {result['confident']}")
        print(f"rag_ms    : {rag_ms:.2f}")
        print()

        print("=== Formatted Output ===")
        print(_format_documents(result))

        print("\n=== Documents ===")
        for i, doc in enumerate(result.get("documents", []), 1):
            print(f"  [{i}] score={doc.get('score', 0):.4f} source={doc.get('source', '?')}")
            print(f"      Q: {doc['q'][:120]}")
            print(f"      A: {doc['a'][:120]}")
        print("\n=== DONE ===\n")

    asyncio.run(_test())
