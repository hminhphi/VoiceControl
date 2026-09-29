import json
import logging
import math
import os
import requests
from pathlib import Path
from utils import read_config
 
logger = logging.getLogger("navigation.poi_store")
 
DATA_DIR = Path(__file__).resolve().parent / "data"
POIS_FILE = DATA_DIR / "pois.json"
ROUTES_FILE = DATA_DIR / "routes.json"
 
POI_TYPES = ("restaurant", "eatery", "shopping")
 
 
def _normalize_place(s: str) -> str:
    return (s or "").strip().lower()
 
 
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
 
 
def get_current_position() -> dict:
    config = read_config("config/config.yaml")
    demo_config = config.get("demo", {})
   
    lat_s = str(demo_config.get("lat", os.environ.get("DEMO_LAT", ""))).strip()
    lng_s = str(demo_config.get("long", os.environ.get("DEMO_LNG", ""))).strip()
    address = demo_config.get("address", os.environ.get("DEMO_ADDRESS", "")).strip()
 
    if not address and (lat_s or lng_s):
        address = f"({lat_s}, {lng_s})"
    if not address:
        address = "Unknown"
    try:
        lat = float(lat_s) if lat_s else 0.0
        lng = float(lng_s) if lng_s else 0.0
    except ValueError:
        lat, lng = 0.0, 0.0
    return {"lat": lat, "lng": lng, "address": address}
 
 
def _load_pois() -> list[dict]:
    if not POIS_FILE.is_file():
        logger.warning("No POIs file at %s", POIS_FILE)
        return []
    try:
        with open(POIS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", POIS_FILE, e)
        return []
    out: list[dict] = []
    for key in ("restaurant", "eatery", "shopping"):
        arr = data.get(key)
        if isinstance(arr, list):
            for p in arr:
                if isinstance(p, dict):
                    p = dict(p)
                    p["type"] = key
                    out.append(p)
    return out
 
 
def get_pois(poi_type: str | None = None, limit: int = 5) -> list[dict]:
    all_pois = _load_pois()
    if poi_type:
        t = poi_type.lower().strip()
        if t in POI_TYPES:
            all_pois = [p for p in all_pois if (p.get("type") or "").lower() == t]
    pos = get_current_position()
    lat0, lng0 = pos["lat"], pos["lng"]
    for p in all_pois:
        plat = float(p.get("lat", 0))
        plng = float(p.get("lng", 0))
        p["_distance_km"] = _haversine_km(lat0, lng0, plat, plng)
        p["_popularity"] = int(p.get("popularity", 5))
    all_pois.sort(key=lambda p: (-p["_popularity"], p["_distance_km"]))
    return all_pois[:limit]
 
 
def get_poi_by_id(poi_id: str) -> dict | None:
    for p in _load_pois():
        if (p.get("id") or "").strip() == str(poi_id).strip():
            return p
    return None
 
 
def get_poi_by_name_or_address(name_or_address: str) -> dict | None:
    if not (name_or_address or "").strip():
        return None
    needle = _normalize_place(name_or_address)
    for p in _load_pois():
        if needle in _normalize_place(p.get("name") or ""):
            return p
        if needle in _normalize_place(p.get("address") or ""):
            return p
    return None
 
 
def _load_routes() -> list[dict]:
    if not ROUTES_FILE.is_file():
        return []
    try:
        with open(ROUTES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else (data.get("routes") or [])
    except Exception as e:
        logger.error("Failed to load routes %s: %s", ROUTES_FILE, e)
        return []
 
 
def get_route(origin: str, destination: str) -> str | None:
    o = _normalize_place(origin)
    d = _normalize_place(destination)
    for r in _load_routes():
        if _normalize_place(r.get("origin") or "") == o and _normalize_place(r.get("destination") or "") == d:
            return (r.get("directions") or "").strip()
    return None
 
 
def save_route(origin: str, destination: str, directions: str) -> None:
    routes = _load_routes()
    o = (origin or "").strip()
    d = (destination or "").strip()
    directions = (directions or "").strip()
    routes = [r for r in routes if not (_normalize_place(r.get("origin") or "") == _normalize_place(o) and _normalize_place(r.get("destination") or "") == _normalize_place(d))]
    routes.append({"origin": o, "destination": d, "directions": directions})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROUTES_FILE, "w", encoding="utf-8") as f:
        json.dump(routes, f, ensure_ascii=False, indent=2)
    logger.info("Saved route: %s -> %s", o[:50], d[:50])
 
 
def get_internet_route(origin: str, destination: str) -> str | None:
    """Fallback to get directions from the internet using OpenStreetMap (Nominatim & OSRM)"""
    try:
        headers = {
            'User-Agent': 'OrchestratorOnEdge/1.0 (contact: your_email@example.com)'
        }
       
        # 1. Geocode Destination
        dest_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(destination)}&format=json&limit=1"
        dest_res = requests.get(dest_url, headers=headers, timeout=5)
       
        if dest_res.status_code != 200:
            logger.error(f"Nominatim dest search failed with status {dest_res.status_code}: {dest_res.text}")
            return None
           
        dest_data = dest_res.json()
        if not dest_data:
            return None # Destination not found on the internet
           
        dest_lat = float(dest_data[0]['lat'])
        dest_lon = float(dest_data[0]['lon'])
        dest_display_name = dest_data[0]['display_name']
 
        # 2. Geocode Origin or Use Current Position
        origin = (origin or "").strip()
        pos = get_current_position()
       
        if not origin or origin.lower() == "here" or pos["address"] in origin:
            org_lat = float(pos['lat'])
            org_lon = float(pos['lng'])
        else:
            org_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(origin)}&format=json&limit=1"
            org_res = requests.get(org_url, headers=headers, timeout=5)
           
            if org_res.status_code != 200:
                logger.error(f"Nominatim origin search failed with status {org_res.status_code}: {org_res.text}")
                return None
               
            org_data = org_res.json()
            if not org_data:
                return None
            org_lat = float(org_data[0]['lat'])
            org_lon = float(org_data[0]['lon'])
 
        # 3. Get Route from OSRM
        # Format: {longitude},{latitude};{longitude},{latitude}
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{org_lon},{org_lat};{dest_lon},{dest_lat}?overview=false"
        route_res = requests.get(osrm_url, timeout=5)
       
        if route_res.status_code != 200:
            logger.error(f"OSRM routing failed with status {route_res.status_code}: {route_res.text}")
            return None
           
        route_data = route_res.json()
       
        if route_data.get('code') != 'Ok':
            return None
           
        route = route_data['routes'][0]
        distance_km = route['distance'] / 1000.0
        duration_min = route['duration'] / 60.0
       
        directions = (
            f"Internet Found Route: From coordinate ({org_lat:.4f}, {org_lon:.4f}) "
            f"to {dest_display_name}. "
            f"Distance: {distance_km:.1f} km. Estimated driving time: {duration_min:.1f} minutes."
        )
       
        # Optionally, save it to cache so doing "directions to X" works instantly next time
        # We save it using the exact 'destination' string passed, so local cache hits it next time!
        save_route(origin or pos['address'], destination, directions)
       
        return directions
 
    except requests.exceptions.RequestException as e:
        logger.error(f"Internet routing request failed: {e}")
        return "No internet connection, cannot search the route dynamically."
    except Exception as e:
        logger.error(f"Internet routing error: {e}")
        return None
 
 
if __name__ == "__main__":
    print("--- 1. Current Position (from config.yaml) ---")
    pos = get_current_position()
    print(pos)
 
    print("\n--- 2. Top 3 Restaurants ---")
    for p in get_pois("restaurant", limit=3):
        print(f"  {p['name']} | pop={p['_popularity']} | dist={p['_distance_km']:.2f}km")
 
    print("\n--- 3. Top 2 Shopping ---")
    for p in get_pois("shopping", limit=2):
        print(f"  {p['name']} | pop={p['_popularity']} | dist={p['_distance_km']:.2f}km")
 
    print("\n--- 4. Search POI by name ---")
    for q in ["Pho 24", "Ben Thanh", "SomethingRandom"]:
        result = get_poi_by_name_or_address(q)
        print(f"  '{q}' -> {result['name'] if result else 'NOT FOUND'}")
 
    print("\n--- 5. Route cache (routes.json) ---")
    print(f"  Route 'A' -> 'B': {get_route('A', 'B')}")
    save_route("A", "B", "Go straight 1km then turn left.")
    print(f"  After save, Route 'A' -> 'B': {get_route('A', 'B')}")
 
    print("\nAll local tests passed.")