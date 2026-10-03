import json
import logging
from uuid import uuid4

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

from context import PerAgentContext
from registry import AgentRegistry
from utils import timeit

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("orchestrator.client")

# Cache agent cards per base URL: avoids one HTTP round-trip per dispatch.
_AGENT_CARD_CACHE: dict = {}
_AGENT_CARD_LOCK = asyncio.Lock()


async def _get_agent_card(httpx_client: httpx.AsyncClient, url: str):
    cached = _AGENT_CARD_CACHE.get(url)
    if cached is not None:
        return cached
    async with _AGENT_CARD_LOCK:
        cached = _AGENT_CARD_CACHE.get(url)
        if cached is not None:
            return cached
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        _AGENT_CARD_CACHE[url] = agent_card
        return agent_card


@timeit
async def send_to_agents_parallel(
    agent_ids: list,
    user_message: str,
    context: PerAgentContext,
    session_id: str,
    registry: AgentRegistry,
):
    # prepare all coroutines
    coros = [
        send_to_agent(agent_id, user_message, context, session_id, registry)
        for agent_id in agent_ids
    ]
    # execute all in parallel
    results = await asyncio.gather(*coros, return_exceptions=True)
    # Optionally handle exceptions here:
    processed_results = []
    for agent_id, result in zip(agent_ids, results):
        if isinstance(result, Exception):
            logger.error(f"Agent {agent_id} failed: {result}")
            processed_results.append({"agent_id": agent_id, "error": str(result)})
        else:
            processed_results.append(result)
    return processed_results


@timeit
async def send_to_agent(
    agent_id: str,
    user_message: str,
    context: PerAgentContext,
    session_id: str,
    registry: AgentRegistry,
    payload_override: str | None = None,
) -> dict:
    url = registry.get_url(agent_id)
    if not url:
        return {}
    logger.info("Sending to agent %s at URL %s", agent_id, url)
    payload_text = payload_override if payload_override is not None else user_message

    async with httpx.AsyncClient(timeout=300.0) as httpx_client:
        agent_card = await _get_agent_card(httpx_client, url)

        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        req = SendMessageRequest(
            id=uuid4().hex,
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"type": "text", "text": payload_text}],
                    "messageId": uuid4().hex,
                }
            )
        )
        
        response = await client.send_message(req)

    agent_result = _extract_dict_from_response(response)
    agent_result['agent_id'] = agent_id
    logger.info(
        "[agent_result] agent_id=%s success=%s message=%r meta=%s",
        agent_id,
        agent_result.get("success"),
        agent_result.get("message"),
        agent_result.get("meta"),
    )
    
    # Optionally add the original JSON string to context, or the dict if you want
    context.add(session_id, agent_id, "assistant", agent_result)
    return agent_result

def _extract_dict_from_response(response) -> dict:
    root = getattr(response, "root", response)
    result = getattr(root, "result", None)
    if not result:
        result = getattr(response, "result", None)
    if not result:
        return {}

    parts = getattr(result, "parts", None)
    if not parts:
        return {}

    for part in parts:
        part_root = getattr(part, "root", part)
        kind = getattr(part_root, "kind", None)
        if kind == "text":
            text = getattr(part_root, "text", "") or ""
            try:
                return json.loads(text)
            except Exception as e:
                logger.error(f"Failed to parse JSON text: {text}, error: {e}")
                return {}
        # If there's any other text attribute
        text_attr = getattr(part_root, "text", None)
        if text_attr:
            try:
                return json.loads(text_attr)
            except Exception as e:
                logger.error(f"Failed to parse JSON text_attr: {text_attr}, error: {e}")
                return {}

    return {}
