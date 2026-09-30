"""
Target Speaker Extraction (TSE) for voice_processing — GPU (PyTorch CUDA).

ClearVoice only ships audio-visual TSE (`AV_MossFormer2_TSE_16K`, needs video),
so for audio-only we do **separation + target selection**:

  1. separate the mixed utterance into 2 streams (MossFormer2_SS_16K), then
  2. pick the stream that best matches the target speaker (embedding similarity
     to the first ~1.5 s of the turn — the person who spoke right after the
     wake word) or, without embeddings, the loudest stream.

Runs on CUDA (works on x86 `torch+cu128` and Jetson `torch-2.8`). Opt-in and
graceful: if ClearVoice is missing the extractor is disabled.

Environment:
  SPEAKER_TSE_ENABLED=0            enable TSE (default: 0)
  TSE_MODEL=MossFormer2_SS_16K     ClearVoice separation model
  TSE_REF_SEC=1.5                  reference window (start of turn) in seconds

ClearVoice models auto-download on first use into ./checkpoints (i.e.
voice_processing/checkpoints); pre-fetch offline with
`scripts/fetch_assets.py tse`.
"""
from __future__ import annotations

import os
import threading

import numpy as np

_extractor = None
_lock = threading.Lock()
_warned = False


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


class TargetSpeakerExtractor:
    def __init__(self):
        from clearvoice import ClearVoice  # lazy
        self.model = (os.environ.get("TSE_MODEL", "MossFormer2_SS_16K") or "MossFormer2_SS_16K").strip()
        self.ref_sec = float(os.environ.get("TSE_REF_SEC", "1.5") or "1.5")
        print(f"[TSE] ClearVoice separation model={self.model} (CUDA)", flush=True)
        self.cv = ClearVoice(task="speech_separation", model_names=[self.model])
        print("[TSE] ready", flush=True)

    def separate(self, wav16k: np.ndarray):
        """Return a list of 16 kHz streams (float32, same length)."""
        x = np.asarray(wav16k, dtype=np.float32).reshape(1, -1)
        out = self.cv(x, False)          # [spk, batch, length]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 2:                # (batch, length) -> single stream
            out = out.reshape(1, *out.shape)
        return [out[i, 0, :].copy() for i in range(out.shape[0])]

    def extract_target(self, wav16k: np.ndarray, analyzer=None) -> np.ndarray:
        """Separate and return the most likely target-speaker stream.

        `analyzer` (speaker.SpeakerAnalyzer) is used for embedding-based
        selection when available; otherwise the loudest stream wins.
        """
        wav16k = np.asarray(wav16k, dtype=np.float32)
        streams = self.separate(wav16k)
        if len(streams) <= 1:
            return streams[0] if streams else wav16k

        n = int(self.ref_sec * 16000)
        ref = wav16k[:min(n, len(wav16k))]
        ref_emb = analyzer.embed(ref) if analyzer is not None else None

        best, best_score = streams[0], -1e9
        for s in streams:
            if ref_emb is not None and getattr(analyzer, "embedding", None) is not None:
                emb = analyzer.embed(s)
                score = analyzer.similarity(ref_emb, emb)
                label = f"sim={score:.3f}"
            else:
                score = _rms(s)          # fallback: loudest stream
                label = f"rms={score:.4f}"
            print(f"[TSE] stream len={len(s)/16000:.2f}s {label}", flush=True)
            if score > best_score:
                best, best_score = s, score
        return best


def get_extractor():
    global _extractor, _warned
    if not _env_bool("SPEAKER_TSE_ENABLED", False):
        return None
    if _extractor is None:
        with _lock:
            if _extractor is None:
                try:
                    _extractor = TargetSpeakerExtractor()
                except Exception as e:
                    if not _warned:
                        print(f"[TSE] disabled ({e})", flush=True)
                        _warned = True
                    return None
    return _extractor
