# Infotainment Agent

Tells jokes (text output) and plays music (WAV files).
Uses **LLM tool-choice** to determine user intent.

## Tools

| Tool | Input | Output |
|---|---|---|
| `tell_joke` | (none) | Random joke as plain text |
| `play_music` | `query` (song title, optional) | URL path to `.wav` file |

## Logic Flow

```
User text
    |
    v
LLM parse_tool_choice()
    |
    +-- tell_joke  --> random joke from jokes.json --> text response
    +-- play_music --> find song by title/id --> WAV URL response
    +-- null       --> fallback message
```

## Data

- `data/jokes.json` -- Text jokes in `{id, text}` format.
- `data/songs.json` -- Song metadata in `{id, title, path}` format.
- `media/songs/` -- WAV audio files served via static mount.

## Configuration (.env)

```env
INFOTAINMENT_PORT=8004
INFOTAINMENT_BASE_URL=http://localhost:8004
```

## Files

| File | Description |
|---|---|
| `media_store.py` | Loads jokes and songs from JSON data files |
| `llm.py` | LLM-based tool-choice parser |
| `agent_executor.py` | Dispatches tell_joke / play_music actions |
| `prompts.py` | System prompt for LLM tool selection |
| `agent_card.py` | Agent card with skill definitions |
| `main.py` | Uvicorn entry point with static file mounts |
