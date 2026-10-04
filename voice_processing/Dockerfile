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
# faster-whisper: GPU-accelerated CTranslate2 backend (fallback when TRT unavailable).
# sherpa-onnx (SenseVoice/Whisper) remains the CPU fallback.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels \
        "openai-whisper==20240927" \
        "transformers>=5.13.0" \
        accelerate \
        safetensors \
        librosa

# faster-whisper (CTranslate2) — GPU Whisper large-v3 with highest accuracy.
# Same install shape as Dockerfile.x86: --no-deps so it does NOT pull the CPU
# onnxruntime and break the force-installed onnxruntime-gpu below; runtime deps
# are installed explicitly. Needs network (PyPI) at build time.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps "faster-whisper>=1.1.0" \
    && python3 -m pip install \
        "ctranslate2>=4.5.0" \
        huggingface_hub \
        tokenizers \
        av \
        tqdm

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
# Keep the image layer slim: wheels/ (build bind-mount), checkpoints/,
# agent_assets/ (models incl. whisper .pt), kokoro *.onnx and I/O dirs are
# all provided at RUNTIME by the ./voice_processing:/app volume mount, so
# copying them would only add ~5 GB of dead weight (the original bundle
# relied on the mount already). COPY --exclude needs the dockerfile:1
# frontend (>=1.19; the old :1.7 pin failed the build instantly).
COPY --exclude=wheels --exclude=checkpoints --exclude=agent_assets \
    --exclude=output --exclude=input_test \
    --exclude='kokoro_tts/*.onnx' --exclude='kokoro_tts/voices-*.bin' \
    voice_processing/ /app/

# ── 8b. whisper_trt + torch2trt (TensorRT Whisper) ────────────
# Built from the sources kept in the repo. whisper_trt loads the local
# large-v3.pt (WHISPER_BASE_MODEL_PATH) and builds the TRT engine on first
# run, caching it under /app/cache/whisper_trt for later runs.
# tensorrt's python module cannot be imported INSIDE the build container
# (libnvos.so is device-only), but torch2trt's setup.py only needs its
# __version__ -> a 1-line stub on PYTHONPATH during the build is enough;
# the CUDAExtension itself compiles/links against the real NvInfer headers
# and libs shipped in l4t-jetpack (verified: builds + installs cleanly).
# At RUNTIME on the Jetson the compose mounts the host tegra/nvidia dirs,
# so the REAL tensorrt module loads and the TRT STT path works.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation setuptools wheel \
    && if [ -d /app/whisper_trt ]; then \
        python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation /app/whisper_trt; \
    fi \
    && if [ -d /app/torch2trt ]; then \
        mkdir -p /tmp/t2tstub \
        && printf '__version__ = "10.0.0"\n' > /tmp/t2tstub/tensorrt.py \
        && PYTHONPATH=/tmp/t2tstub python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation /app/torch2trt \
        && rm -rf /tmp/t2tstub; \
    fi

# ── 8c. Runtime guard: numpy MUST stay 1.x ────────────────────
# clearvoice pins numpy<2 and the stack is validated on 1.26.4; a numpy 2.x
# upgrade at install time is a known runtime breaker (seen with the old code).
# Fail the BUILD instead of failing at runtime.
RUN python3 -c "\
import numpy; \
v = numpy.__version__; \
print('[guard] numpy', v); \
assert v.startswith('1.26'), f'numpy {v} != 1.26.x - do NOT upgrade numpy' \
"

# ── 8d. Restore the GPU onnxruntime module ────────────────────
# silero-vad/openwakeword pull the CPU `onnxruntime` distribution; its files
# OVERWRITE shared files of onnxruntime-gpu AND leave a CPU pybind state with
# a cpython-tagged filename that Python prefers -> the image ends up CPU-only
# (CUDA EP gone) even after a naive reinstall. Uninstall the CPU dist first,
# then force-reinstall the GPU wheel: clean state, CUDA EP back (asserted).
# x86 host voice is unaffected: its ORT build is CPU-only and it never sets
# ONNX_PROVIDER.
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    (python3 -m pip uninstall -y onnxruntime 2>/dev/null || true) \
    && V=$(python3 -m pip show onnxruntime-gpu | sed -n 's/^Version: *//p') \
    && python3 -m pip install --no-index --find-links /tmp/wheels --force-reinstall --no-deps "onnxruntime-gpu==$V" \
    && python3 -c "import onnxruntime as ort; ps=ort.get_available_providers(); print('[ort]', ort.__version__, ps); assert 'CUDAExecutionProvider' in ps, 'CUDA EP missing'"

# ── 9. Optional: local kokoro-onnx (hatchling backend, offline) ─
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    if [ -d /app/kokoro_tts/kokoro-onnx ]; then \
        python3 -m pip install --no-index --find-links /tmp/wheels hatchling editables \
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
