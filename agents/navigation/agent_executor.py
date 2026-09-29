import json
import logging
import os
 
from typing_extensions import override
 
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
 
from llm import parse_intent
from poi_store import (
    get_current_position,
    get_poi_by_name_or_address,
    get_pois,
    get_route,
    save_route,
)
from prompts import INTENT_SYSTEM
 
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("a2a").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
 
logger = logging.getLogger("navigation")
 
SAVE_DIRECTIONS_PREFIX = "SAVE_DIRECTIONS|"
 
 
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
 
 
def _nearest_poi_response(poi_type: str | None, limit: int) -> str:
    pois = get_pois(poi_type=poi_type, limit=limit)
    pos = get_current_position()
    if not pois:
        return f"No {poi_type or 'places'} found near you. Current position: {pos['address']}."
    lines = [f"Near you ({pos['address']}). Top {len(pois)} by popularity:"]
    for i, p in enumerate(pois, 1):
        name = p.get("name", "Unknown")
        desc = (p.get("description") or "").strip()
        directions = (p.get("directions") or "").strip()
        block = f"{i}. {name}"
        if desc:
            block += f"\n   {desc}"
        if directions:
            block += f"\n   To get there: {directions}"
        lines.append(block)
    return "\n\n".join(lines)
 
 
def _directions_response(origin: str | None, destination_name_or_address: str | None) -> str:
    pos = get_current_position()
    origin = (origin or "").strip() or pos["address"]
    if not destination_name_or_address or not destination_name_or_address.strip():
        return "Please say where you want to go (e.g. 'directions to Pho 24' or 'route to Vincom')."
 
    dest = destination_name_or_address.strip()
 
    # 1. Local cache (routes.json)
    stored = get_route(origin, dest)
    if stored:
        return f"[Offline] From {origin} to {dest}\n\n{stored}\n\nNo map UI in this demo; use the address in your navigation app."
 
    # 2. Local POI data (pois.json)
    matched = get_poi_by_name_or_address(dest)
    if matched:
        name = matched.get("name", "")
        addr = matched.get("address", "")
        directions_text = (matched.get("directions") or "").strip()
        directions = directions_text or f"From {pos['address']}, head toward {addr}."
        return f"[Offline] Destination: {name}\nAddress: {addr}\n\n{directions}\n\nNo map UI in this demo; use the address in your navigation app."
 
    # 3. Not found in local data
    return f"Place '{dest}' not found in offline memory."
 
 
def _introduce_response(place_name_or_address: str | None) -> str:
    if not place_name_or_address or not place_name_or_address.strip():
        return "Please say which place you want to know about (e.g. 'tell me about Pho 24')."
    matched = get_poi_by_name_or_address(place_name_or_address.strip())
    if not matched:
        return f"No place found matching '{place_name_or_address.strip()}'. Try a name or address from nearby restaurants or shopping."
    name = matched.get("name", "Unknown")
    addr = matched.get("address", "")
    desc = (matched.get("description") or "").strip()
    if not desc:
        return f"{name}\nAddress: {addr}"
    return f"{name}\nAddress: {addr}\n\n{desc}"
 
 
class NavigationAgentExecutor(AgentExecutor):
    def warm_up(self) -> None:
        get_current_position()
        get_pois(limit=1)
 
    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        async def _out(success: bool, message: str) -> None:
            body = json.dumps({"success": success, "message": message})
            await event_queue.enqueue_event(new_agent_text_message(body))
 
        text = _get_user_text(context)
        if not text.strip():
            await _out(False, "Please ask for nearby places, directions, or to introduce a place.")
            return
 
        if text.strip().startswith(SAVE_DIRECTIONS_PREFIX):
            rest = text.strip()[len(SAVE_DIRECTIONS_PREFIX):].strip()
            try:
                data = json.loads(rest)
                save_route(
                    data.get("origin") or "",
                    data.get("destination") or "",
                    data.get("directions") or "",
                )
                await _out(True, "Directions saved for next time.")
            except Exception as e:
                logger.exception("Save directions parse error: %s", e)
                await _out(False, "Failed to save directions.")
            return
 
        # LLM intent parsing
        intent_data = await parse_intent(text, INTENT_SYSTEM)
        intent = (intent_data.get("intent") or "").strip().lower()
        poi_type = intent_data.get("poi_type")
        origin = intent_data.get("origin_name_or_address")
        dest = intent_data.get("destination_name_or_address")
        limit = intent_data.get("limit")
        if limit is None or not isinstance(limit, int):
            limit = 3
        limit = max(1, min(5, limit))
 
        if intent == "nearest_poi":
            if not poi_type and ("restaurant" in text.lower() or "food" in text.lower() or "eat" in text.lower()):
                poi_type = "restaurant"
            elif not poi_type and ("shop" in text.lower() or "mall" in text.lower()):
                poi_type = "shopping"
            elif not poi_type:
                poi_type = "restaurant"
            out = _nearest_poi_response(poi_type, limit)
        elif intent == "introduce":
            out = _introduce_response(dest)
        elif intent == "directions":
            out = _directions_response(origin, dest)
        else:
            out = _nearest_poi_response("restaurant", 3)

        # Decide success flag based on response content
        success = True
        low = out.lower()
        if (
            "not found in offline memory" in low
            or low.startswith("no place found matching")
            or low.startswith("no ")
            or low.startswith("please say where you want to go")
            or low.startswith("please say which place you want to know about")
        ):
            success = False

        await _out(success, out)
 
    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")
 
 
if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    from dotenv import load_dotenv
 
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
 
    class MockEventQueue:
        async def enqueue_event(self, event):
            try:
                text = event.root.parts[0].text
            except AttributeError:
                try:
                    text = event.parts[0].root.text
                except AttributeError:
                    text = str(event)
            print(f"OUT: {text}")
 
    async def main():
        executor = NavigationAgentExecutor()
        queue = MockEventQueue()
 
        print("\nNavigation Agent CLI (LLM + Internet Routing)")
        executor.warm_up()
 
        while True:
            try:
                user_input = input("\nNavigation > ").strip()
                if user_input.lower() in [":q", "exit", "quit"]:
                    break
                if not user_input:
                    continue
 
                class _Part:
                    def __init__(self, text):
                        self.root = type('obj', (object,), {'kind': 'text', 'text': text})()
 
                class _Msg:
                    def __init__(self, text):
                        self.parts = [_Part(text)]
 
                class _Params:
                    def __init__(self, text):
                        self.message = _Msg(text)
 
                class MockRequest:
                    def __init__(self, text):
                        self.request = type('obj', (object,), {'params': _Params(text)})()
 
                context = MockRequest(user_input)
                await executor.execute(context, queue)
            except KeyboardInterrupt:
                break
 
    asyncio.run(main())