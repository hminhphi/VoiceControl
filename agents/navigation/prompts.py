DESCRIPTION = """Handle navigation requests: find nearby places (restaurants, shops, landmarks), check what is around you, and give simple offline directions or routes to specific destinations from the current location."""

def build_orchestrator_description() -> str:
    return f"""PURPOSE: {DESCRIPTION}"""


INTENT_SYSTEM = """You are a navigation intent classifier. Given the user message, respond with exactly one JSON object (no markdown):
{
  "intent": "nearest_poi" | "directions" | "introduce",
  "poi_type": "restaurant" | "eatery" | "shopping" | null,
  "origin_name_or_address": null or string (if user said "from X" or "from X to Y"),
  "destination_name_or_address": null or string (place name or address for directions/introduce),
  "limit": number 1-5 (for nearest_poi; default 3)
}
- nearest_poi: user wants to go eat / drink / shop without naming a place (e.g. "I want to go eat", "nearest restaurant"). Set poi_type from context (restaurant, shopping).
- directions: user wants routing to a specific place (e.g. "directions to Pho 24", "how to get to Vincom"). Set destination_name_or_address.
- introduce: user wants to know about a place (e.g. "introduce Pho 24", "tell me about Ben Thanh Market"). Set destination_name_or_address to the place name."""
