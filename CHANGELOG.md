# Changelog

All notable changes to this project. Format based on Keep a Changelog;
versions follow `vYYYY.MM.DD`.

## [v2026.11.11] — 2026-11-11

Deployment bundle `orchestrator-on-edge-jetson-20261109.zip`
(commit `3ae2c5b`). **Not yet deployed/verified on the Jetson.**

### Added
- `CHANGELOG.md` (this file).
- Jetson offline model staging: `jetson-offline/llm/gemma-4-E4B-it-Q4_K_M.gguf`
  (MD5 `F4B391DEE3021A70F28A8AAB579688D5`) for llama-cpp on Orin
  (`.env`: `LLM_MODEL_FILE=gemma-4-E4B-it-Q4_K_M.gguf`,
  `LOCAL_LLM_MODEL=gemma-4-E4B-it`).

### Changed
- Release/deploy notes now record artifact MD5s for USB-to-device checks.
- USB layout cleaned: older bundles `20261005` / `20261011` removed to free
  space (rollback via git tags + release zips).

### Investigated / deferred
- **TensorRT-Edge-LLM 0.11.1** as a llama-server replacement: Orin is an
  official target only on JetPack 7.2.x (CUDA 13.2, FP16/INT8/INT4 engines
  built on-device). Current Orin is JetPack 6 / r36.4.0 → deferred until the
  device is reflashed. TREL wheels/checkpoints were removed from the USB.
- **Nemotron Voice Agent blueprint**: architecture reference only — Orin is
  not supported by the blueprint (Thor/JetPack 7 only); its NeMo-Speech.cpp
  speech stack is already the one vendored here.
- **Pipecat**: turn-detection/diarization patterns already mirrored
  (Silero 0.2 s pause → Smart Turn v3, 1.0–1.2 s fallback); no framework
  adoption planned.

### Fixed (in bundle 20261109, commit `3ae2c5b`)
- Diarization never ran on Jetson: compose passed the model index name,
  overriding the bundled GGUF path (offline HF lookup → 20 s timeout per
  turn). Now the bundled GGUF always wins; timeout 8 s; circuit breaker
  after 3 failures (skip for 120 s).
- car_control burned 8.1 s/turn on unreachable AWS AppSync GraphQL:
  `CAR_CONTROL_LOCAL_ONLY=1` (Jetson `.env`) drives the local simulator;
  GraphQL timeouts tightened to 2 s/4 s for PC deployments.
- STT on Orin: `FASTER_WHISPER_BEAM=1` (~2× faster; 2.3 s → ~1.1 s).
- `/state` 404 on the A2A entrypoint: state API now mounted in
  `agents/car_control/main.py`, so the LLM receives live car state.

## [v2026.10.31] — 2026-10-31

Voice stack v3 (commits `5ae9d6e`, `cff4915`, `3ae2c5b`).

### Added
- **Smart Turn v3.2** semantic end-of-turn (pipecat-ai ONNX int8, BSD-2):
  vetoes premature turn end; turn tail 2.5 s → 1.2 s, segment tail 1.0 → 0.8 s.
- **Nemotron-3-Diarization** per turn (NeMo-Speech.cpp GGUF q8): speaker
  count + overlap ratio; overlap gate available.
- **Cross-turn speaker identity**: sherpa-onnx CAM++ embeddings + running
  centroids, persisted in `cache/speakers.json`; roll-call renaming
  ("I am Minh" → name).
- **Streaming STT partials** while the user is still speaking (UI only).
- **Realtime pipeline visualization** in `car_control_ui`: animated
  nine-stage chain, waveform, speaker chip, event ticker; new SVG car art;
  pipeline state endpoint `/api/pipeline` (10 Hz snapshot).
- Arm compose service `car_control_ui`; bundle now ships it (on-device UI).
- Car state + "last touched" hints injected into the LLM system prompt;
  singular door commands operate exactly one door and name it in the reply.

### Changed
- Control replies cut to one short sentence; Turn-2 stream truncated at the
  first sentence for `control_car`.
- Removed the "sending" notification beep (added first-audio latency).
- Docker/ARM and x86 compose/config pass-through for all new switches.

## [v2026.10.11] — 2026-10-11

- Pure-LLM tool-calling orchestrator (router/fast-return removed).
- STT hallucination defenses L0–L4, language gate + script fallback.
- VAD: process_all drain, start gate + 3-frame window, empty-turn re-arm, CPU.
- TTS: speed 1.05, silence trim, sanitize (parens/URL/email), sentence splitter
  fixes, FPT lead-in, think-strip, `max_tokens` 2048.

## [v2026.10.05.1] / [v2026.10.05]

- CTranslate2 built with CUDA for Jetson (PyPI aarch64 wheel is CPU-only).
- AEC/NS fixes eliminating "tách tách" artifacts; faster-whisper default.
