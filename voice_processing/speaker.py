"""
Speaker overlap / diarization / target-speaker selection for voice_processing.

Runs on GPU via PyTorch CUDA (available on both x86 `torch+cu128` and Jetson
`torch-2.8.0 aarch64`), so it stays GPU even where sherpa-onnx is a CPU build.
Opt-in and graceful: if pyannote is missing or the model cannot be loaded
(gated model without token), the analyzer is disabled and the pipeline is
unaffected.

Backend: pyannote.audio
  - diarization pipeline (speaker turns + overlapped speech)
  - speaker embedding for target-speaker selection (enroll from the wake-word)

Environment:
  SPEAKER_ENABLED=0                 enable the stage (default: 0)
  SPEAKER_DEVICE=auto              auto | cuda | cpu (default: auto->cuda)
  SPEAKER_PIPELINE=pyannote/speaker-diarization-community-1
  SPEAKER_EMBEDDING=pyannote/embedding
  SPEAKER_HF_TOKEN=                 token (falls back to HF_TOKEN/HUGGINGFACE_TOKEN)
  SPEAKER_TARGET_THRESHOLD=0.65     cosine sim to accept the enrolled target
  SPEAKER_OVERLAP_MIN=0.15          overlap ratio above which speech is "mixed"

Note: pyannote models are gated on HuggingFace — accept the model conditions and
provide a token. See docs/SPEAKER_OVERLAP.md.
"""
from __future__ import annotations

import os
import threading

import numpy as np

_analyzer = None
_analyzer_lock = threading.Lock()
_warned = False


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _hf_token() -> str:
    return (_env("SPEAKER_HF_TOKEN") or _env("HF_TOKEN")
            or _env("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")).strip()


def find_overlaps(turns):
    """Merge intervals where >= 2 distinct speakers are simultaneously active."""
    events = []
    for s, e, spk in turns:
        events.append((s, +1, spk))
        events.append((e, -1, spk))
    events.sort(key=lambda x: (x[0], x[1]))
    active: dict[str, int] = {}
    overlaps = []
    open_start = None
    for t, delta, spk in events:
        n_before = sum(1 for c in active.values() if c > 0)
        active[spk] = active.get(spk, 0) + delta
        n_after = sum(1 for c in active.values() if c > 0)
        if n_before < 2 <= n_after and open_start is None:
            open_start = t
        elif n_before >= 2 > n_after and open_start is not None:
            overlaps.append((open_start, t))
            open_start = None
    if open_start is not None and events:
        overlaps.append((open_start, events[-1][0]))
    return overlaps


class SpeakerAnalyzer:
    def __init__(self):
        import torch  # noqa
        from pyannote.audio import Pipeline

        device = _env("SPEAKER_DEVICE", "auto").lower() or "auto"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipeline_name = _env("SPEAKER_PIPELINE", "pyannote/speaker-diarization-community-1")
        self.embedding_name = _env("SPEAKER_EMBEDDING", "pyannote/embedding")
        token = _hf_token() or None

        print(f"[SPEAKER] pyannote pipeline={self.pipeline_name} device={self.device} "
              f"token={'yes' if token else 'no'}", flush=True)
        self.pipeline = Pipeline.from_pretrained(self.pipeline_name, token=token)
        self.pipeline.to(torch.device(self.device))

        self.embedding = None
        try:
            from pyannote.audio import Model, Inference
            self._Inference = Inference
            model = Model.from_pretrained(self.embedding_name, token=token)
            self.embedding = Inference(model, window="whole", device=torch.device(self.device))
        except Exception as e:
            print(f"[SPEAKER] embedding model unavailable ({e}); target selection disabled", flush=True)
        self.target_threshold = float(_env("SPEAKER_TARGET_THRESHOLD", "0.65"))
        print("[SPEAKER] ready", flush=True)

    # ── diarization + overlap ───────────────────────────────────────────────
    def analyze(self, waveform_16k: np.ndarray, sample_rate: int = 16000) -> dict:
        import torch
        wav = torch.from_numpy(np.asarray(waveform_16k, dtype=np.float32)).unsqueeze(0)
        out = self.pipeline({"waveform": wav, "sample_rate": sample_rate})
        diar = getattr(out, "speaker_diarization", out)
        turns = [(float(t.start), float(t.end), str(spk))
                 for t, _, spk in diar.itertracks(yield_label=True)]
        turns.sort()
        overlaps = find_overlaps(turns)
        dur = len(waveform_16k) / float(sample_rate)
        # dominant speaker by total speech time
        totals: dict[str, float] = {}
        for s, e, spk in turns:
            totals[spk] = totals.get(spk, 0.0) + (e - s)
        dominant = max(totals, key=totals.get) if totals else None
        return {
            "turns": turns,
            "overlaps": overlaps,
            "dominant": dominant,
            "duration": dur,
            "overlap_ratio": round(sum(e - s for s, e in overlaps) / max(dur, 1e-6), 3),
        }

    # ── target speaker selection ────────────────────────────────────────────
    def embed(self, waveform_16k: np.ndarray, sample_rate: int = 16000):
        if self.embedding is None:
            return None
        import torch
        wav = torch.from_numpy(np.asarray(waveform_16k, dtype=np.float32)).unsqueeze(0)
        emb = self.embedding({"waveform": wav, "sample_rate": sample_rate})
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def similarity(self, emb_a, emb_b) -> float:
        if emb_a is None or emb_b is None:
            return 0.0
        return float(np.dot(emb_a, emb_b))


def get_analyzer():
    """Return a process-wide SpeakerAnalyzer, or None if disabled/unavailable."""
    global _analyzer, _warned
    if not _env_bool("SPEAKER_ENABLED", False):
        return None
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                try:
                    _analyzer = SpeakerAnalyzer()
                except Exception as e:
                    global _warned
                    if not _warned:
                        print(f"[SPEAKER] disabled ({e})", flush=True)
                        _warned = True
                    return None
    return _analyzer


# Mixed-overlap clarification lines (reply in the detected language).
CLARIFY = {
    "en": "I hear several people at once. Could you repeat, please?",
    "ja": "複数の声が聞こえます。もう一度お願いします。",
    "vi": "Tôi nghe nhiều người cùng lúc. Bạn nhắc lại giúp tôi nhé.",
    "ko": "여러 사람 목소리가 들립니다. 다시 말씀해 주세요.",
    "zh": "我听到多人同时说话，请再说一次。",
}


def clarify_message(lang: str) -> str:
    return CLARIFY.get((lang or "en").split("-")[0], CLARIFY["en"])
