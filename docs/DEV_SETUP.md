# Developer Setup (amd64 / Windows / Linux)

This guide is for running and developing on a **dev workstation (x86_64)**.
Deployment to the Jetson is a separate artifact — see
[`DEPLOY.md`](../DEPLOY.md) and `scripts/pack_jetson.*`.

Git tracks **source, config, docs and small custom assets only**. Models,
caches, virtualenvs, secrets and the arm64 wheels are fetched locally — see
[`ASSETS.md`](ASSETS.md).

## 1. Prerequisites

- Python 3.10+ (repo targets 3.10 for containers; host mode uses your local 3.11/3.12)
- [uv](https://docs.astral.sh/uv/) (agent services) and `pip`
- Docker + Docker Compose (for the container workflow)
- Optional: NVIDIA GPU + recent driver (for the amd64 GPU stack)
- A local LLM server exposing an OpenAI-compatible API (`llama-server`, Ollama, …)

## 2. Clone & fetch assets

```bash
git clone <repo-url> orchestrator-on-edge
cd orchestrator-on-edge

# Linux/macOS
scripts/fetch_assets.sh --all
# Windows PowerShell
.\scripts\fetch_assets.ps1 --all
```

This downloads the GGUF, embedding model, Kokoro, sherpa ASR/KWS and the arm64
wheels, then runs `verify`. See [`ASSETS.md`](ASSETS.md) for the full map.

## 3. Configure

```bash
cp .env.x86.example .env.x86     # amd64 / Docker workflow
cp .env.example     .env         # Jetson-style local runs / run_all.sh
```

Fill in the secrets you need (`GRAPHQL_*`, `OPENAI_API_KEY`, …). Both `.env`
files are gitignored.

## 4. Run — Docker (amd64 + NVIDIA GPU)

```bash
# Build the shared base once
docker build -f Dockerfile.x86-base -t orchestrator-on-edge/x86-base:cu128 .

# Bring up the stack
docker compose -f docker-compose.x86.yml --env-file .env.x86 up -d
```

- Orchestrator: http://localhost:8000 (`/agents`, `/health`, `/docs`)
- Car simulator UI: http://localhost:8010
- Frontend (optional profile): http://localhost:3000

Makefile shortcuts: `make build-x86-base`, `make x86-up`, `make x86-down`,
`make x86-logs`, `make x86-download-models`.

## 5. Run — Host mode on Windows (stub car + real mic)

`run_all_pc.ps1` starts the backend on host Python with hardware stubs from
`stubs/` (no Jetson hardware required) and a real microphone via
`voice_processing`:

```powershell
.\run_all_pc.ps1                 # real microphone
.\run_all_pc.ps1 -VoiceStub      # keyboard text instead of mic (stubs/voice_stub_main.py)
.\run_all_pc.ps1 -Docker         # backend in Docker, mic on host
.\run_all_pc.ps1 -Down           # stop everything
```

It uses `.env.x86` and isolates each service in its own window with a
`[(service)]` title.

## 6. Local LLM server

Either point `LOCAL_LLM_URL` at an existing OpenAI-compatible server, or run
`llama-server` with the fetched GGUF:

```bash
llama-server -m llama-cpp/models/Qwen3.5-4B-Q4_K_M.gguf \
  --jinja --host 0.0.0.0 --port 8080 -ngl 99 -fa
```

Docker services reach it via `host.docker.internal:8080`.

## 7. Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/agents
curl -X POST http://localhost:8000/v1/orchestrator/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Turn on the headlights", "session_id": "dev"}'
```

Integration tests: `python test/run_tests.py`.

## 8. Repackage for the Jetson

Once assets are present and you want to ship the arm64 artifact:

```bash
scripts/pack_jetson.sh          # Linux/macOS
.\scripts\pack_jetson.ps1       # Windows
# -> dist/orchestrator-on-edge-jetson-<YYYYMMDD>.zip
```

The zip is self-contained (arm64 code + config + models, `.env.example` only)
and excludes all x86/dev-only files. See [`DEPLOY.md`](../DEPLOY.md).
