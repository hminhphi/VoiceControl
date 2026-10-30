"""Cross-turn speaker identity — sherpa-onnx speaker embeddings.

Nemotron's diarization slots are utterance-local (they restart per call), so a
persistent "who is this" needs a voiceprint: every turn is embedded
(CAM++ / 3D-Speaker, 192-d, 27 MB ONNX, CPU) and matched against running
centroids. A match above the threshold keeps the existing label; otherwise a
new speaker is created (up to `SPEAKER_ID_MAX_SPEAKERS`).

Env:
  SPEAKER_ID_ENABLED=0         enable the stage
  SPEAKER_ID_MODEL             ONNX path
  SPEAKER_ID_THRESHOLD=0.65    cosine similarity to keep the same speaker
  SPEAKER_ID_MAX_SPEAKERS=8
  SPEAKER_ID_STORE             JSON file with centroids (default cache/speakers.json)
"""
import json
import re
import os
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(
    _HERE, "agent_assets", "models", "speaker", "campplus_16k.onnx"
)
DEFAULT_STORE = os.path.join(_HERE, "cache", "speakers.json")
SAMPLE_RATE = 16000


def _env(name, default=""):
    return os.environ.get(name) or default


class SpeakerIdentifier:
    def __init__(self, model_path=None, threshold=None, max_speakers=None, store=None):
        import sherpa_onnx

        self.model_path = model_path or _env("SPEAKER_ID_MODEL", DEFAULT_MODEL)
        self.threshold = float(
            threshold if threshold is not None else _env("SPEAKER_ID_THRESHOLD", "0.65")
        )
        self.max_speakers = int(
            max_speakers if max_speakers is not None else _env("SPEAKER_ID_MAX_SPEAKERS", "8")
        )
        self.store_path = store or _env("SPEAKER_ID_STORE", DEFAULT_STORE)
        self.available = False
        self.error = None
        try:
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.model_path,
                num_threads=2,
                provider="cpu",
                debug=False,
            )
            if not cfg.validate():
                raise RuntimeError(f"invalid speaker embedding model: {self.model_path}")
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self.available = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.extractor = None
        self._centroids = []  # [(label, np.ndarray)]
        self._lock = threading.Lock()
        if self.available:
            self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self):
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._centroids = [
                (item["label"], np.asarray(item["embedding"], dtype=np.float32))
                for item in data.get("speakers", [])
            ]
        except FileNotFoundError:
            self._centroids = []
        except Exception as e:
            print(f"[SPK-ID] store ignored ({e})", flush=True)
            self._centroids = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            payload = {
                "speakers": [
                    {"label": label, "embedding": emb.tolist()}
                    for label, emb in self._centroids
                ]
            }
            tmp = self.store_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, self.store_path)
        except Exception as e:
            print(f"[SPK-ID] store write failed: {e}", flush=True)

    def rename(self, old_label, new_label):
        """Give a diarization slot a human name (spoken roll call)."""
        with self._lock:
            for i, (lab, emb) in enumerate(self._centroids):
                if lab != old_label:
                    continue
                if new_label == lab:
                    return False
                self._centroids[i] = (new_label, emb)
                break
            else:
                return False
        self._save()
        return True

    # ── api ──────────────────────────────────────────────────────────────
    def embed(self, wav_16k):
        if not self.available:
            return None
        audio = np.asarray(wav_16k, dtype=np.float32)
        if audio.size < SAMPLE_RATE // 2:
            return None
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak * 0.9
        stream = self.extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        vec = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None

    def identify(self, wav_16k, name=None):
        vec = self.embed(wav_16k)
        if vec is None:
            return None
        with self._lock:
            best = None
            best_score = -1.0
            for label, centroid in self._centroids:
                score = float(np.dot(vec, centroid))
                if score > best_score:
                    best, best_score = label, score
            if best is not None and best_score >= self.threshold:
                idx = next(i for i, (lab, _) in enumerate(self._centroids) if lab == best)
                label, centroid = self._centroids[idx]
                # Slow centroid update so one odd turn cannot drift the voiceprint.
                merged = (0.9 * centroid + 0.1 * vec).astype(np.float32)
                self._centroids[idx] = (label, merged / np.linalg.norm(merged))
                is_new = False
            elif len(self._centroids) < self.max_speakers:
                label = name or f"Speaker {len(self._centroids) + 1}"
                self._centroids.append((label, vec))
                best_score = 1.0
                is_new = True
            else:
                label, centroid = max(
                    self._centroids,
                    key=lambda item: float(np.dot(vec, item[1])),
                )
                best_score = float(np.dot(vec, centroid))
                is_new = False
            self._save()
        return {
            "id": label,
            "label": label,
            "score": round(best_score, 3),
            "is_new": is_new,
            "known_speakers": len(self._centroids),
        }


_SINGLETON = None
_LOCK = threading.Lock()


ROLLCALL_RE = re.compile(
    r"(?i)\b(?:t[ôo]i t[êe]n l[àa]|t[ôo]i l[àa]|m[ìi]nh l[àa]|h[ôo] t[ôo]i l[àa]"
    r"|my name is|i am|i\x27m)\s+([^\W\d_]{2,20})"
)


def rollcall_name(text):
    """Name when the user introduces themselves ("I am Minh")."""
    m = ROLLCALL_RE.search(text or "")
    if not m:
        return None
    name = m.group(1).strip().split()
    return (name[0].capitalize() if name else None)


def get_identifier():
    global _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            _SINGLETON = SpeakerIdentifier()
            if _SINGLETON.available:
                print(
                    f"[SPK-ID] speaker embeddings ready "
                    f"(model={os.path.basename(_SINGLETON.model_path)}, "
                    f"threshold={_SINGLETON.threshold}, "
                    f"known={len(_SINGLETON._centroids)})",
                    flush=True,
                )
            else:
                print(f"[SPK-ID] unavailable: {_SINGLETON.error}", flush=True)
        return _SINGLETON
