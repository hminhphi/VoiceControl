.PHONY: blue llm warmup docker-up run build-base build-voice download-models

# ── Shared base image ─────────────────────────────────────────────────────────
# Build once; all GPU services inherit from this image.
# Run from project root where Dockerfile.l4t-base lives.
L4T_BASE_TAG := orchestrator-on-edge/l4t-base:r36.4.0-torch2.8

build-base:
	docker build \
		-f Dockerfile.l4t-base \
		-t $(L4T_BASE_TAG) \
		.

# Build only the voice_processing image (depends on build-base first)
build-voice: build-base
	docker compose build voice_processing

# Download sherpa-onnx models onto the Jetson (run inside voice_processing container or directly)
download-models:
	docker compose run --rm voice_processing python3 download_models.py

# ── x86 / amd64 + NVIDIA GPU targets ─────────────────────────────────────────
X86_BASE_TAG := orchestrator-on-edge/x86-base:cu126
X86_COMPOSE  := docker compose -f docker-compose.x86.yml

build-x86-base:
	docker build \
		-f Dockerfile.x86-base \
		-t $(X86_BASE_TAG) \
		.

build-x86-voice: build-x86-base
	$(X86_COMPOSE) build voice_processing

build-x86-all: build-x86-base
	$(X86_COMPOSE) build

x86-up: build-x86-base
	$(X86_COMPOSE) up -d

x86-down:
	$(X86_COMPOSE) down

x86-logs:
	$(X86_COMPOSE) logs -f voice_processing

x86-download-models:
	$(X86_COMPOSE) run --rm voice_processing python3 download_models.py

x86-test:
	$(X86_COMPOSE) run --rm voice_processing python3 test_voice_pipeline.py --skip-stt


BLUE_REPO := /home/acevn/ai-demo-edge-main

# 1) Connect Bluetooth speaker using local setup_blue.sh
blue:
	chmod +x ./setup_blue.sh
	./setup_blue.sh

# 2) Start LLM server using existing demo repo (assumes its Makefile/run.sh handle details)
llm:
	$(MAKE) -C $(BLUE_REPO) run

# 3) Send a few warmup requests to the LLM HTTP API (llama-server on :8080)
warmup:
	sleep 10
	@echo "Warming up LLM (1/3)..."
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{ "model": "llama", "messages": [ {"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"Say hello briefly."} ] }' >/dev/null || true
	@echo "Warming up LLM (2/3)..."
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{ "model": "llama", "messages": [ {"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"What is 2 plus 2?"} ] }' >/dev/null || true
	@echo "Warming up LLM (3/3)..."
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{ "model": "llama", "messages": [ {"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"Give me a very short fun fact."} ] }' >/dev/null || true

# 4) Bring up full edge stack
docker-up:
	docker compose -p orch_v1 up

# Full sequence: BT -> LLM -> warmup -> wait -> docker compose up
run: blue
	( $(MAKE) -C $(BLUE_REPO) run ) &
	$(MAKE) warmup
	sleep 30
	docker compose -p orch_v1 up

