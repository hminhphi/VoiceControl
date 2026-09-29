import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("navigation.llm")

LOCAL_LLM_URL_DEFAULT = "http://host.docker.internal:11434/v1"


def _client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return AsyncOpenAI(api_key=api_key, timeout=30.0)
    local_url = os.environ.get("LOCAL_LLM_URL", LOCAL_LLM_URL_DEFAULT).strip()
    return AsyncOpenAI(base_url=local_url, api_key="not-needed", timeout=30.0)


def _model() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return os.environ.get("LOCAL_LLM_MODEL", "").strip() or "llama3.2"


async def parse_intent(user_message: str, system_prompt: str) -> dict[str, Any]:
    client = _client()
    model = _model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        return json.loads(raw) if raw else {}
    except Exception as e:
        logger.error("LLM intent parse failed: %s", e)
        return {}


if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    async def test():
        from prompts import INTENT_SYSTEM

        test_messages = [
            "directions to Pho 24",
            "nearest restaurant",
            "tell me about Ben Thanh Market",
        ]
        for msg in test_messages:
            print(f"\nInput: {msg}")
            result = await parse_intent(msg, INTENT_SYSTEM)
            print(f"Output: {json.dumps(result, indent=2)}")

    asyncio.run(test())
