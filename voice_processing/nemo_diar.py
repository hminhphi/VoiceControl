"""Per-turn speaker diarization — NVIDIA Nemotron-3-Diarization.

Runs the NeMo-Speech.cpp `diarize` command on the finished turn audio and
returns how many people spoke, their segments and the overlap ratio (two
speakers active at the same time). This is the safety signal for the meeting
demo: when two people talk over each other we do not act on the command.

The GGUF model (q8_0, 102 MB, OpenMDW-1.1) and the runtime binary are shipped
in `agent_assets/models/nemo` and `vendor/nemo_speech`.

Env:
  NEMO_DIAR_ENABLED=0      enable the stage
  NEMO_SPEECH_BIN          runtime path (default vendor/nemo_speech/nemo-speech)
  NEMO_SPEECH_DIAR_MODEL   GGUF path or indexed model name
  NEMO_DIAR_DEVICE=cpu     cpu | cuda | vulkan
  NEMO_DIAR_PRESET         v3-offline (default) | v3-streaming
  NEMO_DIAR_TIMEOUT_SEC=20 hard timeout for one call

Cross-turn identity is handled by `speaker_id.py` (sherpa-onnx embeddings):
Nemotron's speaker slots are utterance-local, so a persistent id needs the
embedding clustering on top.
"""
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(_HERE, "vendor", "nemo_speech", "nemo-speech")
if os.name == "nt":
    DEFAULT_BIN = os.path.join(_HERE, "vendor", "nemo_speech", "win", "nemo-speech.exe")

# On Linux the runtime ships as a tarball (its shared libraries are symlinks,
# which cannot live on an NTFS bind mount). It is extracted once into the
# writable cache dir and reused from there.
_RUNTIMES = {
    ("Linux", "x86_64"): os.path.join(
        _HERE, "agent_assets", "nemo_speech", "nemo-speech-0.3.0-linux-x86_64-cpu.tar.gz"
    ),
    ("Linux", "aarch64"): os.path.join(
        _HERE, "agent_assets", "nemo_speech", "nemo-speech-0.3.0-linux-aarch64-cpu.tar.gz"
    ),
    ("Darwin", "arm64"): os.path.join(
        _HERE, "agent_assets", "nemo_speech", "nemo-speech-0.3.0-macos-aarch64-cpu.tar.gz"
    ),
}
# POSIX: use the container-native tmp (its filesystem supports symlinks; a
# Windows bind mount does not, so extraction there would break lib*.so links).
_EXTRACT_ROOT = os.path.join(tempfile.gettempdir(), "nemo_speech")
SAMPLE_RATE = 16000


def _env(name, default=""):
    return os.environ.get(name) or default


def _env_float(name, default):
    try:
        raw = os.environ.get(name)
        return float(raw) if raw not in (None, "") else float(default)
    except ValueError:
        return float(default)


def _binary():  # noqa: N802
    return _resolve_binary()


DEFAULT_MODEL = os.path.join(
    _HERE, "agent_assets", "models", "nemo", "diarization.q8_0.gguf"
)


def _extracted_runtime():
    """Extract the platform tarball into the cache dir; return its binary."""
    tarball = _RUNTIMES.get((platform.system(), platform.machine()))
    if not tarball or not os.path.isfile(tarball):
        return None
    try:
        with tarfile.open(tarball) as tf:
            names = [n for n in tf.getnames() if n.endswith("/bin/nemo-speech")]
            if not names:
                return None
            root = names[0].rsplit("/bin/", 1)[0]
            if not os.path.isdir(os.path.join(_EXTRACT_ROOT, root)):
                os.makedirs(_EXTRACT_ROOT, exist_ok=True)
                tf.extractall(_EXTRACT_ROOT)
        bin_path = os.path.join(_EXTRACT_ROOT, root, "bin", "nemo-speech")
        if not os.path.isfile(bin_path):
            return None
        os.chmod(bin_path, 0o755)
        return bin_path
    except Exception as e:
        print(f"[DIAR] runtime extraction failed: {e}", flush=True)
        return None


def _resolve_binary():
    env = _env("NEMO_SPEECH_BIN")
    if env:
        path = env
        if os.name == "nt" and not path.lower().endswith(".exe") and os.path.isfile(path + ".exe"):
            path = path + ".exe"
        return path
    if os.name == "nt":
        local = os.path.join(_HERE, "vendor", "nemo_speech", "win", "nemo-speech.exe")
        if os.path.isfile(local):
            return local
    local = os.path.join(_HERE, "vendor", "nemo_speech", "nemo-speech")
    if os.path.isfile(local):
        return local
    return _extracted_runtime()


def _model():
    env = _env("NEMO_SPEECH_DIAR_MODEL")
    # Compose always passes an index name by default; prefer the bundled
    # GGUF so the runtime never attempts an (offline) HF download.
    if os.path.isfile(DEFAULT_MODEL):
        return DEFAULT_MODEL
    if env:
        return env
    return "nemotron-3-diarization"


def _overlap_ratio(segments, total):
    """Fraction of the turn where two or more speakers are active at once."""
    if not segments or total <= 0:
        return 0.0
    events = []
    for seg in segments:
        events.append((float(seg["start"]), 1))
        events.append((float(seg["end"]), -1))
    events.sort()
    active = 0
    prev = events[0][0]
    overlap = 0.0
    for t, delta in events:
        if active >= 2:
            overlap += max(0.0, min(t, total) - max(prev, 0.0))
        prev = t
        active += delta
    return max(0.0, min(1.0, overlap / total))


class NemoDiarizer:
    def __init__(self, binary=None, model=None):
        self.binary = binary or _binary()
        self.model = model or _model()
        self.device = _env("NEMO_DIAR_DEVICE", "cpu")
        self.preset = _env("NEMO_DIAR_PRESET", "v3-offline")
        self.timeout = _env_float("NEMO_DIAR_TIMEOUT_SEC", 8.0)
        self.available = os.path.isfile(self.binary)
        # Circuit breaker: a failing runtime must not cost time per turn.
        self._disabled_until = 0.0
        self._fail_count = 0

    def analyze(self, wav_16k):
        """Diarize one turn. `wav_16k` is float32 mono 16 kHz."""
        if not self.available:
            return None
        if time.time() < self._disabled_until:
            return None
        audio = np.asarray(wav_16k, dtype=np.float32)
        if audio.size < SAMPLE_RATE // 2:
            return None
        total = audio.size / SAMPLE_RATE
        pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        fd, path = tempfile.mkstemp(prefix="diar_", suffix=".wav")
        os.close(fd)
        try:
            import soundfile as sf

            sf.write(path, pcm, SAMPLE_RATE, subtype="PCM_16")
            cmd = [
                self.binary, "diarize", path,
                "--model", self.model,
                "--device", self.device,
                "--preset", self.preset,
                "--format", "json",
                "--quiet",
            ]
            t0 = time.time()
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, cwd=_HERE
            )
            latency = (time.time() - t0) * 1000.0
            if out.returncode != 0:
                err = (out.stderr or out.stdout or "").strip().splitlines()
                raise RuntimeError(err[-1] if err else f"exit {out.returncode}")
            payload = json.loads(out.stdout[out.stdout.index("{"):])
            segments = payload.get("segments") or []
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._fail_count = 0
        self._disabled_until = 0.0
        speakers = sorted({int(s["speaker"]) for s in segments if s.get("speaker") is not None})
        duration = float(payload.get("duration") or total)
        return {
            "n_speakers": len(speakers),
            "speakers": speakers,
            "segments": segments,
            "overlap_ratio": round(_overlap_ratio(segments, duration), 3),
            "speech_ratio": round(
                min(1.0, sum(float(s["end"]) - float(s["start"]) for s in segments) / max(duration, 1e-6)),
                3,
            ),
            "latency_ms": round(latency, 1),
            "backend": "nemotron-3-diarization",
        }


_SINGLETON = None
_LOCK = threading.Lock()


def get_diarizer():
    global _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            _SINGLETON = NemoDiarizer()
            if not _SINGLETON.available:
                print(
                    f"[DIAR] runtime not found: {_SINGLETON.binary} (diarization disabled)",
                    flush=True,
                )
            else:
                print(
                    f"[DIAR] nemotron-3-diarization ready model={_SINGLETON.model} "
                    f"device={_SINGLETON.device}",
                    flush=True,
                )
        return _SINGLETON


def note_failure(max_count=3, hold=120.0):
    """Circuit-break the diarizer after repeated failures (offline box).
    Call from the caller except block.
    """
    global _SINGLETON
    d = _SINGLETON
    if d is None:
        return
    d._fail_count += 1
    if d._fail_count >= max_count:
        d._disabled_until = time.time() + hold
        print(f"[DIAR] disabled for {hold:.0f}s after {d._fail_count} failures", flush=True)

def speaker_label(spk):
    if spk is None:
        return "unknown"
    return f"Speaker {int(spk) + 1}"
