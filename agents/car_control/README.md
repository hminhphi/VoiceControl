# Car Control Agent

Controls non-safety-critical car hardware via JSON commands.
**Does not use LLM** -- parses JSON directly and dispatches to the simulator.

## Components and Actions

| Component | Actions | Description |
|---|---|---|
| `door` | `open`, `close`, `lock`, `unlock` | Car door |
| `window` | `open`, `close` | Power window |
| `trunk` | `open`, `close` | Trunk / tailgate |
| `light` | `on`, `off` | Headlights |
| `ac` | `on`, `off` | Air conditioning |
| `mirror` | `fold`, `unfold` | Side mirrors |

## Input Format

```json
{"component": "door", "action": "open"}
{"component": "light", "action": "on"}
{"component": "ac", "action": "off"}
{"component": "mirror", "action": "fold"}
```

## Output Format

```json
{"success": true, "message": "The door is open."}
{"success": false, "message": "No connection to the car."}
```

## Logic Flow

1. Receive text from orchestrator.
2. Match one of the six demo actions by exact phrase or explicit command pattern.
3. Dispatch the matching GraphQL command when GraphQL is configured.
4. Return status JSON with the matched action and parser metadata.

Embedding fallback is disabled by default because nearest-neighbor matching can turn unrelated STT transcripts into a supported action. Set `CAR_CONTROL_ENABLE_EMBEDDING_MATCH=true` only for local experiments.

## Configuration (.env)

```env
CAR_CONTROL_PORT=8001
CAR_CONTROL_BASE_URL=http://localhost:8001
CAR_SIMULATOR_CONNECTED=true
```

## Files

| File | Description |
|---|---|
| `car_simulator.py` | Legacy in-memory simulator |
| `agent_executor.py` | Six-command door/trunk matcher and GraphQL dispatcher |
| `agent_card.py` | Agent card with skill definitions per component |
| `prompts.py` | Routing description for orchestrator |
| `main.py` | Uvicorn entry point |
