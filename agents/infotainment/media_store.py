import json
import logging
import random
from pathlib import Path

logger = logging.getLogger("infotainment.media_store")

DATA_DIR = Path(__file__).resolve().parent / "data"
JOKES_FILE = DATA_DIR / "jokes.json"
SONGS_FILE = DATA_DIR / "songs.json"


def _load_jokes() -> list[dict]:
    if not JOKES_FILE.is_file():
        return []
    try:
        with open(JOKES_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to load jokes: %s", e)
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict) and (x.get("text") or x.get("id"))]
    return data.get("jokes") or []


def _load_songs() -> list[dict]:
    if not SONGS_FILE.is_file():
        return []
    try:
        with open(SONGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else (data.get("songs") or [])
    except Exception as e:
        logger.error("Failed to load songs: %s", e)
        return []


def get_random_joke() -> dict | None:
    """Return a random joke dict with 'text' field, or None."""
    jokes = _load_jokes()
    if not jokes:
        return None
    return random.choice(jokes)


def get_song_by_title_or_id(query: str) -> dict | None:
    """Find a song by title or id (case-insensitive substring match)."""
    query = (query or "").strip().lower()
    if not query:
        # No specific query -> return a random song
        songs = _load_songs()
        return random.choice(songs) if songs else None
    songs = _load_songs()
    for s in songs:
        if query in (s.get("title") or "").lower() or query in (s.get("id") or "").lower():
            return s
    return None


def get_random_song() -> dict | None:
    """Return a random song from the library."""
    songs = _load_songs()
    return random.choice(songs) if songs else None


def get_song_titles() -> list[str]:
    return [s.get("title") or s.get("id") or "" for s in _load_songs() if s.get("title") or s.get("id")]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("--- Media Store Test ---")
    
    joke = get_random_joke()
    print(f"Random Joke: {joke.get('text') if joke else 'None'}")
    
    song = get_random_song()
    print(f"Random Song: {song.get('title') if song else 'None'}")
    
    titles = get_song_titles()
    print(f"All Song Titles: {titles}")

