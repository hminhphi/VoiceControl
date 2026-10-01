# syntax=docker/dockerfile:1
# ============================================================
# voice_processing — Voice AI service (Jetson AGX)
#
# Inherits from the shared l4t-base image which already contains:
#   PyTorch 2.8.0, ONNX Runtime GPU 1.23.0, sherpa-onnx, soxr, numpy
#
# FULLY OFFLINE BUILD: every pip install resolves from the local
# wheelhouse bind-mounted at /tmp/wheels (= voice_processing/wheels
# in the build context; copy it from the USB before building).
#   --no-index --find-links /tmp/wheels   -> nothing fetched from PyPI
#   --no-build-isolation                  -> use the base image's
#       setuptools (<70) so classic-setup.py sdists (PyAudio, pesq,
#       mojimoji) build without network access.
#
# NOTE: the frontend is `docker/dockerfile:1` (>=1.19) on purpose:
# COPY --exclude needs frontend 1.19+. The old pin `docker/dockerfile:1.7`
# made the build fail immediately ("unknown flag: --exclude").
# ============================================================
ARG BASE_IMAGE=orchestrator-on-edge/l4t-base:r36.4.0-torch2.8
FROM ${BASE_IMAGE}

ENV PYTHONPATH=/app
ENV XDG_CACHE_HOME=/app/cache

WORKDIR /app

# ── 1. Audio-specific system packages (offline: vendor/apt-debs) ─
# (base already has portaudio, alsa, pulseaudio, ffmpeg; this covers
# libasound2-plugins when the base was built online-less.)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=bind,source=vendor/apt-debs,target=/tmp/apt-debs,readonly \
    if ls /tmp/apt-debs/*.deb >/dev/null 2>&1; then \
        dpkg -i /tmp/apt-debs/*.deb || true; \
    else \
        apt-get update && apt-get install -y --no-install-recommends \
            libasound2-plugins \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# ── 2. Audio Python packages ──────────────────────────────────
# NOTE: no python-dotenv — runtime reads container env directly via
# os.environ (no .env loading in voice_processing/main.py).
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation \
        sounddevice==0.5.1 \
        PyAudio \
        pydub \
        colorlog \
        requests \
        websocket-client

# ── 3. silero-vad (VAD) ───────────────────────────────────────
# NOTE: no webrtcvad — VAD is Silero ONNX (vad.py); AEC is WebRTC APM
# native lib (libs/webrtc_apm), neither imports the webrtcvad package.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels \
        silero-vad==5.1.2

# ── 4. Wake word (openwakeword kept as fallback) ──────────────
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels \
        openwakeword==0.5.1

# ── 5. STT backends ───────────────────────────────────────────
# STT_BACKEND=whisper → TensorRT (whisper_trt) on Jetson, same model as x86
# (WHISPER_MODEL=large-v3, loaded from agent_assets/models/whisper/large-v3.pt).
# sherpa-onnx (SenseVoice/Whisper) remains the CPU fallback.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels \
        "openai-whisper==20240927" \
        "transformers>=5.13.0" \
        accelerate \
        safetensors \
        librosa

# ── 6. TTS (Kokoro ONNX) ──────────────────────────────────────
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels \
        "espeakng-loader>=0.2.4" \
        "phonemizer-fork>=3.3.2"

# Japanese G2P for Kokoro's ja voice (misaki JAG2P -> fugashi/MeCab).
# pyopenjtalk-plus ships aarch64 wheels; unidic-lite comes as a built wheel.
# (The plain `misaki[ja]` extra pulls `unidic`, whose dict is NOT auto-
# downloaded, so fugashi fails and Kokoro falls back to espeak - wrong ja
# phonemes, "weird" Japanese.)
# Cython MUST be installed first: mojimoji ships a .pyx and builds without
# isolation, so it cythonizes with the env's Cython.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation \
        Cython \
    && python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation \
        "misaki" \
        "fugashi" \
        "jaconv" \
        "mojimoji" \
        "pyopenjtalk-plus" \
        "unidic-lite"

# ── 7. Speaker overlap / diarization (opt-in, GPU via torch CUDA) ─────
# pyannote.audio runs on the torch CUDA already present in l4t-base.
# Models are gated on HuggingFace: accept the model conditions and set
# SPEAKER_HF_TOKEN (or HF_TOKEN) at runtime.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels \
        "pyannote.audio>=3.1.0" clearvoice

# ── 8. Application source ─────────────────────────────────────
# wheels/ and checkpoints/ stay OUT of the image layer (they are only
# needed as bind-mounts during build; the runtime volume-mount shadows
# /app anyway). COPY --exclude needs the dockerfile:1 frontend.
COPY --exclude=wheels --exclude=checkpoints voice_processing/ /app/

# ── 8b. torch2trt + whisper_trt (TensorRT Whisper) ────────────
# Built from the sources kept in the repo. whisper_trt loads the local
# large-v3.pt (WHISPER_BASE_MODEL_PATH) and builds the TRT engine on first
# run, caching it under /app/cache/whisper_trt for later runs.
# Skipped gracefully (STT falls back to sherpa-whisper) when the TensorRT
# python module is not importable in this image.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation setuptools wheel \
    && if [ -d /app/torch2trt ] && python3 -c "import tensorrt" 2>/dev/null; then \
        python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation /app/torch2trt /app/whisper_trt; \
    else echo "[setup] tensorrt python not available; skip whisper_trt (sherpa fallback)"; fi

# ── 9. Optional: local kokoro-onnx (hatchling backend, offline) ─
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -d /app/kokoro_tts/kokoro-onnx ]; then \
        python3 -m pip install --no-index --find-links /tmp/wheels hatchling \
        && python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation -e /app/kokoro_tts/kokoro-onnx; \
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
