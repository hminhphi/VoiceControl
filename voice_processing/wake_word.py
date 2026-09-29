"""
Wake word detector.

Hỗ trợ 2 backends (chọn qua WAKE_WORD_BACKEND):
  - "openwakeword" (default, legacy): openwakeword ONNX — tương thích ngược
  - "sherpa_onnx":                    sherpa-onnx KeywordSpotter — chính xác hơn
                                       trong tiếng ồn, hỗ trợ custom keywords

Environment variables:
  WAKE_WORD_BACKEND      openwakeword | sherpa_onnx  (default: openwakeword)
  WAKE_WORD_THRESHOLD    float 0-1  (default: 0.5)
  WAKE_WORD_MODEL_DIR    path to sherpa-onnx keyword spotter model dir
                         (must contain encoder.onnx, decoder.onnx, joiner.onnx,
                          tokens.txt, keywords.txt)
"""

import os
import time
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_ROOT, "agent_assets", "models")
OWW_CHUNK = 1280    # openwakeword frame size at 16 kHz
SHERPA_CHUNK = 512  # sherpa-onnx KeywordSpotter frame size at 16 kHz
SAMPLE_RATE = 16000

# Legacy openwakeword model paths
OWW_MODELS = [
    os.path.join(ASSETS, "hey_doh_ra.onnx"),
    os.path.join(ASSETS, "hey_dola.onnx"),
]


def _model_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ── Backend: openwakeword (legacy) ────────────────────────────────────────────

class _OWWBackend:
    """openwakeword-based wake word detection (original behaviour)."""

    def __init__(self, detect_prob: float = 0.5):
        import openwakeword

        existing = [p for p in OWW_MODELS if os.path.isfile(p)]
        for p in OWW_MODELS:
            name = _model_basename(p)
            status = "loaded" if p in existing else "MISSING"
            print(f"[wake_word][oww] {name}: {status}", flush=True)
        if not existing:
            raise FileNotFoundError(f"No openwakeword models under {ASSETS}")

        self.detect_prob = detect_prob
        self.model = openwakeword.Model(wakeword_models=existing, inference_framework="onnx")
        self._frame_length = OWW_CHUNK
        print(f"[wake_word][oww] Ready. threshold={detect_prob}", flush=True)

    @property
    def frame_length(self) -> int:
        return self._frame_length

    def predict(self, audio_int16: np.ndarray) -> tuple[bool, str, float]:
        """Return (detected, model_name, prob)."""
        pred = self.model.predict(audio_int16)
        if isinstance(pred, dict) and pred:
            best_key, prob = max(pred.items(), key=lambda x: x[1])
            name = _model_basename(best_key) if ("/" in best_key or "\\" in best_key) else best_key
            return prob >= self.detect_prob, name, float(prob)
        prob = float(pred) if pred is not None else 0.0
        return prob >= self.detect_prob, "wakeword", prob


# ── Backend: sherpa-onnx KeywordSpotter ──────────────────────────────────────

class _SherpaBackend:
    """
    sherpa-onnx KeywordSpotter — better noise robustness than openwakeword.

    Model dir must contain:
      encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt, keywords.txt
    """

    def __init__(self, model_dir: str, detect_threshold: float = 0.2):
        import sherpa_onnx

        encoder  = os.path.join(model_dir, "encoder.onnx")
        decoder  = os.path.join(model_dir, "decoder.onnx")
        joiner   = os.path.join(model_dir, "joiner.onnx")
        tokens   = os.path.join(model_dir, "tokens.txt")
        keywords = os.path.join(model_dir, "keywords.txt")

        for f in (encoder, decoder, joiner, tokens, keywords):
            if not os.path.isfile(f):
                raise FileNotFoundError(f"[wake_word][sherpa] missing: {f}")

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords,
            num_threads=int(os.environ.get("WAKE_WORD_THREADS", "2")),
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            max_active_paths=2,
            keywords_score=1.8,
            keywords_threshold=detect_threshold,
            num_trailing_blanks=1,
            provider="cpu",
        )
        self._stream = self._spotter.create_stream()
        self._frame_length = SHERPA_CHUNK
        self._cooldown = 1.5
        self._last_detection = 0.0
        print(
            f"[wake_word][sherpa] Ready | model_dir={model_dir} "
            f"threshold={detect_threshold}",
            flush=True,
        )

    @property
    def frame_length(self) -> int:
        return self._frame_length

    def predict(self, audio_int16: np.ndarray) -> tuple[bool, str, float]:
        """Return (detected, keyword, score)."""
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        self._stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=audio_f32)

        if not self._spotter.is_ready(self._stream):
            return False, "", 0.0

        self._spotter.decode_stream(self._stream)
        result = self._spotter.get_result(self._stream)

        if result:
            now = time.time()
            # Anti-repeat cooldown
            if now - self._last_detection < self._cooldown:
                self._spotter.reset_stream(self._stream)
                return False, str(result), 1.0
            self._last_detection = now
            self._spotter.reset_stream(self._stream)
            return True, str(result), 1.0

        return False, "", 0.0

    def __del__(self):
        try:
            del self._spotter
            del self._stream
        except Exception:
            pass


# ── WakeWordProcessor (public API, unchanged interface) ───────────────────────

class WakeWordProcessor:
    """
    Drop-in replacement for the original WakeWordProcessor.

    Selects backend based on WAKE_WORD_BACKEND env var.
    Internal resampler converts from device_sample_rate → 16 kHz.
    """

    def __init__(self, device_sample_rate: int, detect_prob: float = 0.5):
        self.device_sample_rate = device_sample_rate
        self.detect_prob = detect_prob
        self._detected = False
        self._last_model_name: str | None = None
        self._last_prob: float = 0.0

        backend_name = os.environ.get("WAKE_WORD_BACKEND", "openwakeword").strip().lower()
        print(f"[wake_word] Backend: {backend_name}", flush=True)

        if backend_name in ("sherpa_onnx", "sherpa", "kws"):
            model_dir = os.environ.get(
                "WAKE_WORD_MODEL_DIR",
                os.path.join(ASSETS, "kws"),
            )
            self._backend = _SherpaBackend(model_dir, detect_threshold=detect_prob)
        else:
            self._backend = _OWWBackend(detect_prob=detect_prob)

        self._frame_length = self._backend.frame_length
        self.buffer = np.array([], dtype=np.int16)
        self.resampler = None

    def update_state(self, audio_bytes: bytes) -> None:
        """Feed raw audio bytes (device sample rate, int16) into the detector."""
        chunk = np.frombuffer(audio_bytes, dtype=np.int16)

        # Resample to 16 kHz if needed
        if self.resampler is None:
            from resampler import Resampler
            self.resampler = Resampler(self.device_sample_rate, SAMPLE_RATE)

        chunk_f32 = chunk.astype(np.float32) / 32768.0
        resampled_f32 = self.resampler.resample(chunk_f32)
        resampled_int16 = (resampled_f32 * 32767).astype(np.int16)

        self.buffer = np.concatenate((self.buffer, resampled_int16))
        if self._detected:
            return

        while len(self.buffer) >= self._frame_length:
            frame = self.buffer[: self._frame_length].copy()
            self.buffer = self.buffer[self._frame_length:]
            detected, name, prob = self._backend.predict(frame)
            self._last_prob = prob
            if detected:
                self._last_model_name = name
                self._detected = True
                print(
                    f"[wake_word] Detected: {name!r} prob={prob:.3f}",
                    flush=True,
                )
                break

    def update_state_f32(self, audio_f32: np.ndarray) -> None:
        """Feed AEC-processed float32 audio (already at device rate)."""
        pseudo_bytes = (audio_f32 * 32768).astype(np.int16).tobytes()
        self.update_state(pseudo_bytes)

    def get_detected(self) -> bool:
        """Return True once on detection, then reset."""
        out = self._detected
        self._detected = False
        self._last_model_name = None
        self._last_prob = 0.0
        return out
