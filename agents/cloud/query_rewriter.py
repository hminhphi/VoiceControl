import asyncio
import json
import logging
import os
import urllib.error
import urllib.request


logger = logging.getLogger("cloud.query_rewriter")

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

REWRITE_SYSTEM_PROMPT = """Rewrite this ASR transcript into the most likely clean English search query.
Only fix speech recognition errors.
Do not answer.
Do not invent facts.
Preserve role acronyms like CEO/CFO/CIO/CAO unless clearly dotted.
If uncertain, return the original.

Prioritize automotive-domain company/executive queries, including CEO, CFO, CIO, CTO, and other corporate officer titles.
Known car brands include Mitsubishi, Toyota, Nissan, Honda, DENSO.
Use very high priority for short executive queries like "who is CIO of Mitsu/Mitsubishi" and "who is CFO of Nissan".

For Mitsubishi, treat close ASR mishearings as Mitsubishi when the query is asking about an automotive company or executive.
Examples that should rewrite to Mitsubishi include: Mitsu, miss subishi, mr b.c, mr. b.c, mr. obc, mistel, mizu.
Do not rewrite these Mitsubishi-like sounds to another brand.
Do not switch to Toyota, Honda, or DENSO unless that other brand is clearly present in the transcript.
If the brand sound is ambiguous but close to Mitsubishi/Mitsu in an executive query, prefer Mitsubishi over inventing or choosing another brand.

Return JSON only: {"query": "...", "changed": true/false, "confidence": 0.0-1.0}"""


def _parse_rewrite_payload(content: str, original_query: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("[query_rewrite] invalid JSON from LLM: %r", content)
        return {
            "query": original_query,
            "changed": False,
            "confidence": 0.0,
            "error": "invalid_json",
        }

    rewritten_query = str(data.get("query") or original_query).strip() or original_query
    changed = bool(data.get("changed")) and rewritten_query != original_query
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "query": rewritten_query,
        "changed": changed,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _rewrite_query_sync(query: str, model: str, api_key: str, timeout: float) -> dict:
    body = {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }
    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return _parse_rewrite_payload(content, query)


async def rewrite_query_with_llm(query: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4.1-mini"
    timeout = float(os.environ.get("QUERY_REWRITE_TIMEOUT", "6.0"))

    if not api_key:
        return {
            "query": query,
            "changed": False,
            "confidence": 0.0,
            "model": model,
            "skipped": "missing_openai_api_key",
        }

    try:
        result = await asyncio.to_thread(_rewrite_query_sync, query, model, api_key, timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.warning("[query_rewrite] HTTP error status=%s body=%s", exc.code, body[:500])
        return {
            "query": query,
            "changed": False,
            "confidence": 0.0,
            "model": model,
            "error": f"http_{exc.code}",
        }
    except Exception as exc:
        logger.warning("[query_rewrite] failed: %s", exc)
        return {
            "query": query,
            "changed": False,
            "confidence": 0.0,
            "model": model,
            "error": exc.__class__.__name__,
        }

    result["model"] = model
    return result
