# Car Control Agent

## 1. Folder structure

```
agents/car_control/
├── main.py              # A2A app, port 8001, /sounds/resolve, static /static/sounds
├── agent_executor.py    # CarControlAgentExecutor
├── agent_card.py        # build_agent_card()
├── prompts.py
├── pyproject.toml
└── Dockerfile
```

(Sounds are mounted from `shared/car_control/` in Docker.)

## 2. Logic flow

(No LLM; demo control supports only left/right door and trunk open/close.)

1. Get user text from context (message parts, kind=text).
2. Match exact demo phrases first.
3. If no exact match, parse explicit command patterns where the action is adjacent to a supported target.
4. If no explicit command is found, return `{ "success": false, "message": "" }`.
5. If matched, dispatch the corresponding GraphQL command when GraphQL is configured.
6. Return JSON `{ "success": true, "message": [matched action], "meta": { "match_source": ... } }`.

Embedding fallback is disabled by default for safety. Set `CAR_CONTROL_ENABLE_EMBEDDING_MATCH=true` only for experiments; it should not be used as the authority for physical controls.
