# Infotainment Agent

## 1. Folder structure

```
agents/infotainment/
├── main.py             # mount /static/jokes, /static/songs
├── agent_executor.py   # InfotainmentAgentExecutor
├── agent_card.py
├── prompts.py         # TOOL_SYSTEM (LLM tool descriptions)
├── llm.py             # parse_tool_choice(user_message, system_prompt)
├── media_store.py     # get_random_joke, get_song_by_title_or_id, get_song_titles
├── data/
│   ├── jokes.json     # id, path (.wav)
│   └── songs.json     # id, title, path
├── pyproject.toml
└── Dockerfile
```

(Media files live under shared/infotainment/jokes/, shared/infotainment/songs/, mounted in Docker.)

## 2. Logic flow

(Call LLM to choose tool; then invoke one of two tools: tell_joke, play_music.)

1. Get user text from context.
2. If text empty → return _out(false, "Say 'tell a joke' or 'play music'.").
3. **Call LLM** parse_tool_choice(text, TOOL_SYSTEM) → JSON with "tool" ("tell_joke" | "play_music" | null) and optional "query" (for play_music), "message" (if tool is null).
4. If tool == "tell_joke" → **tool tell_joke**: get_random_joke(); if none → _out(false, "No jokes in library."); else build joke URL, return _out(true, "Here's a joke.", is_play_sound=true, sound_path=url).
5. If tool == "play_music" → **tool play_music**: get_song_by_title_or_id(query or text); if no song → return _out with suggestion list or "No songs"; else build song URL, return _out(true, "Playing: {title}", is_play_sound=true, sound_path=url).
6. Else (tool null or unknown) → return _out(false, LLM message or "Say 'tell a joke' or 'play music'.").
