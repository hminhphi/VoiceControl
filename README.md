# Orchestrator on Edge

An AI-powered in-car assistant demo that runs entirely on edge devices. The system uses a multi-agent architecture where a central orchestrator routes user requests to specialized agents, each handling a different domain (vehicle control, manual lookup, navigation, infotainment, cloud search).

All agents communicate via the [A2A (Agent-to-Agent) protocol](https://github.com/google/A2A) and use a local LLM served via an OpenAI-compatible API (e.g. [llama-server](https://github.com/ggerganov/llama.cpp), Ollama, or OpenAI).

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Option A: Docker Compose (recommended)](#option-a-docker-compose-recommended)
  - [Option B: Run natively without Docker](#option-b-run-natively-without-docker)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Platform Notes](#platform-notes)


---

## Architecture

```
                          POST /message
                               |
                               v
                      +----------------+
                      |  Orchestrator  |  :8000
                      |  (FastAPI)     |
                      +-------+--------+
                              |
          LLM routing         |  A2A protocol
      (llama-server :8082)    |
                              |
        +-----------+---------+---------+-----------+
        |           |                   |           |
        v           v                   v           v
  +-----------+ +-----------+   +-----------+ +-----------+
  |car_control| |car_manual |   |navigation | |infotain-  |
  |  :8001    | |  :8002    |   |  :8003    | |ment :8004 |
  +-----------+ +-----------+   +-----------+ +-----------+
                                                    |
                                              +-----------+
                                              |  cloud    |
                                              |  :8005    |
                                              +-----------+
```

**How it works:**

1. User sends a natural language message to the Orchestrator (`POST /message`).
2. The Orchestrator uses an LLM to analyze intent and route the message to the correct agent(s).
3. Each agent processes the request and returns a JSON response.
4. The Orchestrator merges results and sends a final response back to the user.

---

## Project Structure

```
orchestrator-on-edge/
|-- .env                        # All configuration in one file
|-- docker-compose.yml          # Docker deployment for all services
|-- orchestrator/               # Central routing service (FastAPI)
|   |-- main.py                 # FastAPI app, POST /message, GET /agents
|   |-- router.py               # Routes user message to agent(s)
|   |-- llm.py                  # LLM client (OpenAI-compatible API)
|   |-- registry.py             # Agent discovery via A2A agent cards
|   |-- client.py               # Sends messages to agents via A2A
|   |-- context.py              # Per-session conversation context
|   +-- prompts.py              # Routing system prompt templates
|-- agents/
|   |-- car_control/            # Controls car components (door, window, AC, etc.)
|   |   |-- agent_executor.py   #   Parses {component, action} JSON
|   |   +-- car_simulator.py    #   Simulates car hardware state
|   |-- car_manual/             # Answers questions from car manual (RAG)
|   |   |-- agent_executor.py   #   QAStore + optional LLM synthesis
|   |   |-- qa_store.py         #   BM25 + embedding hybrid search
|   |   |-- interactive_test.py #   Standalone REPL for testing
|   |   +-- data/               #   Manual Q&A data (toyota.json, mercedes.json)
|   |-- navigation/             # Nearby POIs, directions, place info
|   |   |-- agent_executor.py   #   Intent: nearest_poi, directions, introduce
|   |   +-- poi_store.py        #   POI data and route storage
|   |-- infotainment/           # Music playback and jokes
|   |   |-- agent_executor.py   #   Tool routing: play_music, tell_joke
|   |   +-- media_store.py      #   Song and joke data
|   +-- cloud/                  # Simulated cloud search (directions lookup)
|       +-- directions_store.py #   Dummy directions data
|-- shared/                     # Shared assets (sounds, media files)
|-- test/                       # Integration test scripts
|   |-- run_tests.py            # Sends requests to orchestrator, checks results
|   |-- test_routing_single.json
|   +-- test_flow_complex.json
```

Each agent follows the same pattern:
- `main.py` -- starts the A2A HTTP server (uvicorn)
- `agent_card.py` -- defines the agent's identity, skills, and routing hints
- `agent_executor.py` -- implements the business logic
- `prompts.py` -- agent-specific prompt templates

---

## Prerequisites

- **Python 3.11+**
- **uv** (Python package manager): `pip install uv`
- **LLM server** with an OpenAI-compatible API (e.g. llama-server, Ollama) and a GGUF model
- **Docker and Docker Compose** (if using Docker deployment)

---

## Quick Start

> **Developers:** see [`docs/DEV_SETUP.md`](docs/DEV_SETUP.md) for the full
> amd64/Windows workflow (Docker or host mode + stubs) and
> [`docs/ASSETS.md`](docs/ASSETS.md) for fetching models. Deployment to the
> Jetson is described in [`DEPLOY.md`](DEPLOY.md).

### Option A: Docker Compose (recommended)

**Step 1: Clone and configure**

```bash
git clone <repo-url> orchestrator-on-edge
cd orchestrator-on-edge

cp .env.x86.example .env.x86     # amd64 / Docker development
cp .env.example .env             # or the base template
# fetch models once (GGUF, embedding, Kokoro, sherpa, wheels)
scripts/fetch_assets.sh --all    # Linux/macOS
.\scripts\fetch_assets.ps1 --all # Windows
```

Edit `.env` to set `LLM_PORT` to match your LLM server port (default: 8082).

**Step 2: Start the LLM server**

```bash
llama-server -m /path/to/your-model.gguf --port 8082 --host 0.0.0.0
```

**Step 3: Build and run all services**

```bash
docker compose up --build
```

Wait for all 6 services to start. The orchestrator will retry connecting to agents
until they are ready.

**Step 4: Verify**

```bash
# List registered agents
curl http://localhost:8000/agents

# Send a message
curl -X POST http://localhost:8000/message \
    -H "Content-Type: application/json" \
    -d '{"message": "How do I open the trunk?"}'
```

Open `http://localhost:8000/docs` in a browser for the interactive Swagger UI.

---

### Option B: Run natively without Docker

This is useful for development, debugging, or when Docker is not available.

**Step 1: Configure**

```bash
cd orchestrator-on-edge
```

Edit `.env`:
```
LOCAL_LLM_URL=http://localhost:8082/v1
AGENT_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004,http://localhost:8005
```

**Step 2: Start the LLM server**

```bash
llama-server -m /path/to/your-model.gguf --port 8082 --host 0.0.0.0
```

**Step 3: Start each agent (one terminal per agent)**

```bash
# Terminal 1
cd agents/car_control && uv sync && uv run python main.py

# Terminal 2
cd agents/car_manual && uv sync && uv run python main.py

# Terminal 3
cd agents/navigation && uv sync && uv run python main.py

# Terminal 4
cd agents/infotainment && uv sync && uv run python main.py

# Terminal 5
cd agents/cloud && uv sync && uv run python main.py
```

**Step 4: Start the orchestrator (after all agents are running)**

```bash
# Terminal 6
cd orchestrator && uv sync && uv run python main.py
```

**Step 5: Verify** (same as Docker)

```bash
curl http://localhost:8000/agents
curl -X POST http://localhost:8000/message \
    -H "Content-Type: application/json" \
    -d '{"message": "Turn on the headlights"}'
```

---

## Configuration

All configuration is in the `.env` file at the project root. Key variables:

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PORT` | `8082` | LLM server port. Docker Compose reads this to construct the internal URL. |
| `LOCAL_LLM_URL` | `http://localhost:8082/v1` | Full LLM URL for native (non-Docker) runs. |
| `LOCAL_LLM_MODEL` | `qwen3` | Model name passed to the OpenAI-compatible API. |
| `OPENAI_API_KEY` | (empty) | If set, uses OpenAI cloud API instead of local LLM. |
| `OPENAI_MODEL` | `gpt-4o-nano` | OpenAI model name (only used if API key is set). |

### Agent Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_URLS` | `http://localhost:8001` | Comma-separated agent URLs (native runs). Docker Compose overrides this. |
| `CAR_MANUAL_BRAND` | `toyota` | Which car manual data to load (`toyota` or `mercedes`). |
| `CAR_MANUAL_SEARCH_MODE` | `hybrid` | Search mode: `bm25`, `embedding`, or `hybrid`. |
| `CAR_MANUAL_SCORE_THRESHOLD` | `0.3` | Below this score, returns multiple documents instead of one. |

### Docker vs Native

When running with Docker Compose, the `LOCAL_LLM_URL` in `.env` is **ignored** for
containers. Instead, `docker-compose.yml` constructs the URL using
`host.docker.internal:${LLM_PORT}`. The `extra_hosts` directive ensures this works
on Linux and Jetson, not just Windows/macOS.

When running natively, agents read `LOCAL_LLM_URL` directly from `.env`.

---

## API Reference

### Orchestrator (port 8000)

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| POST | `/message` | `{"message": "...", "session_id": "..."}` | Send a user message. Returns `{"success": bool, "message": "..."}`. |
| GET | `/agents` | -- | List all registered agents. |
| POST | `/agents` | `{"url": "http://..."}` | Register a new agent by its base URL. |
| DELETE | `/agents/{agent_id}` | -- | Remove an agent. |

### Individual Agent (A2A protocol)

Each agent exposes:
- `GET /.well-known/agent-card.json` -- agent identity and capabilities
- `POST /` -- A2A JSON-RPC endpoint for `message/send`

Example direct agent call:
```bash
curl -X POST http://localhost:8002/ \
    -H "Content-Type: application/json" \
    -d '{
      "jsonrpc": "2.0",
      "id": "1",
      "method": "message/send",
      "params": {
        "message": {
          "messageId": "msg-001",
          "role": "user",
          "parts": [{"kind": "text", "text": "How to pair bluetooth?"}]
        }
      }
    }'
```

---

## Testing

### Level 1: Unit test (no server needed)

Test the car_manual RAG pipeline directly:

```bash
cd agents/car_manual
uv sync
uv run python interactive_test.py --brand toyota --mode hybrid
```

This opens a REPL where you type questions and get immediate results with timing.

### Level 2: Single agent test

Start one agent and call it directly:

```bash
# Start the agent
cd agents/car_manual && uv run python main.py

# In another terminal, check the agent card
curl http://localhost:8002/.well-known/agent-card.json

# Send a question via A2A
curl -X POST http://localhost:8002/ \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{"messageId":"m1","role":"user","parts":[{"kind":"text","text":"tire pressure warning reset"}]}}}'
```

### Level 3: End-to-end integration test

Start the full system, then run the test suite:

```bash
# Start everything (Docker or native)
docker compose up --build

# Run tests
python test/run_tests.py
```

Or test manually:

```bash
# Manual question -> should route to car_manual
curl -X POST http://localhost:8000/message \
    -H "Content-Type: application/json" \
    -d '{"message": "How do I open the trunk?", "session_id": "test"}'

# Control command -> should route to car_control
curl -X POST http://localhost:8000/message \
    -H "Content-Type: application/json" \
    -d '{"message": "Turn on the headlights", "session_id": "test"}'

# Navigation -> should route to navigation
curl -X POST http://localhost:8000/message \
    -H "Content-Type: application/json" \
    -d '{"message": "Find nearby restaurants", "session_id": "test"}'

# Entertainment -> should route to infotainment
curl -X POST http://localhost:8000/message \
    -H "Content-Type: application/json" \
    -d '{"message": "Tell me a joke", "session_id": "test"}'
```

---

## Platform Notes

### Windows

- Docker Desktop provides `host.docker.internal` automatically.
- Native runs use `localhost` from `.env`.
- No special setup needed.

### Linux (x86/ARM)

- Docker Compose uses `extra_hosts: host.docker.internal:host-gateway` so LLM
  connectivity works the same as on Windows.
- Native runs use `localhost` from `.env`.

### NVIDIA Jetson (Nano / Xavier / Orin)

- Same as Linux. Docker and native runs both work.
- For GPU acceleration, build llama-server with CUDA support
  (see [llama.cpp build instructions](https://github.com/ggerganov/llama.cpp#build)).
- Use `-ngl 99` to offload all layers to GPU when starting llama-server.
- Jetson Nano (4GB RAM): increase swap if the embedding model causes OOM:
  ```bash
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  ```

---

