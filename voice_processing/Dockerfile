# syntax=docker/dockerfile:1.7
# ============================================================
# voice_processing — Voice AI service (Jetson AGX)
#
# Inherits from the shared l4t-base image which already contains:
#   PyTorch 2.8.0, ONNX Runtime GPU 1.23.0, sherpa-onnx, soxr, numpy
#
# This Dockerfile only adds the audio-specific libraries that are
# NOT needed by other services (openwakeword, kokoro-onnx, silero-vad).
# STT is sherpa-onnx only (from l4t-base) — see step 5.
# ============================================================
ARG BASE_IMAGE=orchestrator-on-edge/l4t-base:r36.4.0-torch2.8
FROM ${BASE_IMAGE}

ENV PYTHONPATH=/app
ENV XDG_CACHE_HOME=/app/cache

WORKDIR /app

# ── 1. Audio-specific system packages ─────────────────────────
# (base already has portaudio, alsa, pulseaudio, ffmpeg)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Audio Python packages ──────────────────────────────────
# NOTE: no python-dotenv — runtime reads container env directly via
# os.environ (no .env loading in voice_processing/main.py).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        sounddevice==0.5.1 \
        PyAudio \
        pydub \
        colorlog \
        requests \
        websocket-client

# ── 3. silero-vad (VAD) ───────────────────────────────────────
# NOTE: no webrtcvad — VAD is Silero ONNX (vad.py); AEC is WebRTC APM
# native lib (libs/webrtc_apm), neither imports the webrtcvad package.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        silero-vad==5.1.2

# ── 4. Wake word (openwakeword kept as fallback) ──────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        openwakeword==0.5.1

# ── 5. STT backends ───────────────────────────────────────────
# sherpa-onnx only (STT_BACKEND=sherpa_onnx in .env).
# REMOVED 2026-09: openai-whisper, transformers, accelerate, safetensors,
# librosa (served whisper_trt/nemotron backends — unused in any live path,
# ~1.5-2GB). stt.py keeps lazy imports + auto-fallback to sherpa_onnx, so
# re-adding later is just restoring the pip lines below.
#   openai-whisper==20240927 + transformers>=5.13.0, accelerate,
#   safetensors, librosa

# ── 6. TTS (Kokoro ONNX) ──────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        "espeakng-loader>=0.2.4" \
        "phonemizer-fork>=3.3.2"

# ── 7. Speaker overlap / diarization (opt-in, GPU via torch CUDA) ─────
# pyannote.audio runs on the torch CUDA already present in l4t-base.
# Models are gated on HuggingFace: accept the model conditions and set
# SPEAKER_HF_TOKEN (or HF_TOKEN) at runtime.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "pyannote.audio>=3.1.0"

# ── 8. torch2trt + whisper_trt ─────────────────────────────────
# REMOVED 2026-09: only needed to build whisper_trt engines (unused —
# STT_BACKEND=sherpa_onnx). Sources kept in repo (~1MB) for reference.

# ── 8. Application source ─────────────────────────────────────
# Bind-mount context strips __pycache__, cache/, output/ via .dockerignore.
# wheels/ (303MB) is excluded: pip installs from bind-mounted wheels at
# build time, and the runtime volume-mount shadows /app anyway.
COPY --exclude=wheels voice_processing/ /app/

# ── 9. Optional: local kokoro-onnx ────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -d /app/kokoro_tts/kokoro-onnx ]; then \
        python3 -m pip install -e /app/kokoro_tts/kokoro-onnx; \
    fi

# ── 10. Pre-extract Silero ONNX model asset ──────────────────
RUN mkdir -p /app/agent_assets/models \
    && python3 -c "\
import silero_vad, os, shutil; \
src = os.path.join(os.path.dirname(silero_vad.__file__), 'data', 'silero_vad.onnx'); \
dest = '/app/agent_assets/models/silero_vad.onnx'; \
shutil.copy(src, dest); \
print('[setup] silero_vad.onnx →', dest)"

# ── 11. Download sherpa-onnx models at build time (optional) ─
# Set SHERPA_DOWNLOAD_MODELS=1 to pre-bake models into image.
# Leave unset to download at container startup via download_models.py.
ARG SHERPA_DOWNLOAD_MODELS=0
RUN if [ "$SHERPA_DOWNLOAD_MODELS" = "1" ] && [ -f /app/download_models.py ]; then \
        python3 /app/download_models.py; \
    fi

CMD ["python3", "main.py"]
