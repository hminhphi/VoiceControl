# Navigation Agent

## 1. Folder structure

```
agents/navigation/
├── main.py
├── agent_executor.py   # NavigationAgentExecutor
├── agent_card.py
├── prompts.py         # INTENT_SYSTEM
├── llm.py             # parse_intent(text, system)
├── poi_store.py       # get_current_position (from env), get_pois, get_poi_by_name_or_address, get_route, save_route, get_internet_route
├── data/
│   └── pois.json      # restaurant[] and shopping[]: id, name, address, lat, lng, popularity (1–10), directions, description
├── pyproject.toml
└── Dockerfile
```

Current position is read only from .env: DEMO_LAT, DEMO_LNG, DEMO_ADDRESS (not in dummy data).

## 2. Logic flow

(Step 4 calls LLM; steps 3 and 6–9 use POI/routes data.)

1. Get user text from context.
2. If text empty → return _out(false, "Please ask for nearby places, directions, or to introduce a place.").
3. If text starts with "SAVE_DIRECTIONS|" → parse JSON (origin, destination, directions), call save_route(...), return _out(True/False) accordingly.
4. **Call LLM** parse_intent(text, INTENT_SYSTEM) → intent (nearest_poi | directions | introduce), poi_type, origin, destination_name_or_address, limit.
5. Normalize limit (1–5), infer poi_type from text if missing (e.g. restaurant, shopping).
6. If intent == "nearest_poi" → get_current_position() from env; get_pois(poi_type, limit) sorted by popularity (desc) then distance; format top 3 with name, description, directions → out.
7. If intent == "directions" → 3-tier lookup:
   - (a) get_route(origin, dest) from local cache (routes.json) → if found, return [Offline] directions.
   - (b) get_poi_by_name_or_address(dest) from local POI data (pois.json) → if found, return [Offline] directions.
   - (c) get_internet_route(origin, dest) via Nominatim geocoding + OSRM routing → if found, return [Internet] directions and auto-save to routes.json for next time.
   - (d) If all fail → return "No internet connection, place not in offline memory."
8. If intent == "introduce" → get_poi_by_name_or_address(dest); return name, address, description.
9. Else → out = _nearest_poi_response("restaurant", 3).
10. Return _out(true, out) (JSON { success, message }).

Use cases: (1) User asks for routing to a place → local first, then internet fallback. (2) User asks to introduce a place → response: description. (3) User says e.g. "I want to go eat" (no specific place) → top 3 nearest restaurants sorted by popularity (1–10), with name, description, and directions.
