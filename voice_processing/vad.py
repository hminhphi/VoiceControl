"""
VAD (Voice Activity Detection) processor.

Nâng cấp từ bản gốc (stateless) lên stateful dual-threshold,
inspired by xiaozhi-esp32-server/silero.py.

Cải tiến:
  - Trạng thái h/c (GRU hidden state) được giữ xuyên suốt, thay vì reset theo chunk
  - Dual-threshold: hysteresis giúp giảm false positive trong tiếng ồn
  - Sliding window (frame_window_threshold) trước khi trigger "speech"

Environment vars:
  VAD_THRESHOLD_HIGH  float 0-1  (default 0.55) - ngưỡng bắt đầu speech
  VAD_THRESHOLD_LOW   float 0-1  (default 0.20) - ngưỡng kết thúc speech (hysteresis)
"""

import os
import numpy as np
import onnxruntime
from resampler import Resampler

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_ROOT, "agent_assets", "models", "silero_vad.onnx")
MODEL_CHUNK = 512       # silero v5 chunk size at 16 kHz
VAD_SAMPLE_RATE = 16000
_WINDOW_SIZE = 3        # frames in sliding window before triggering


class VADProcessor:
    """
    Silero VAD v5 ONNX — stateful, dual-threshold.

    Process flow per chunk:
      1. Resample device audio → 16 kHz float32
      2. Feed 512-sample chunks through ONNX model (stateful h/c)
      3. Apply dual-threshold + sliding window → speech probability
    """

    def __init__(self, device_sample_rate: int):
        print(f"[VAD] Model path: {MODEL_PATH}", flush=True)
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"VAD model not found: {MODEL_PATH}")

        self.device_sample_rate = device_sample_rate

        # Threshold from env or defaults
        self._threshold_high = float(os.environ.get("VAD_THRESHOLD_HIGH", "0.55"))
        self._threshold_low = float(os.environ.get("VAD_THRESHOLD_LOW", "0.20"))

        # ONNX session — CPU provider is sufficient for tiny silero model
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

        # Persistent GRU hidden state (2, 1, 128) and context (1, 64)
        self._h = np.zeros((2, 1, 128), dtype=np.float32)
        self._c = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

        # Sliding window for hysteresis
        self._window: list[bool] = []
        self._last_is_voice: bool = False
        self._latest_prob: float = 0.0

        self.buffer = np.array([], dtype=np.float32)
        self.resampler: Resampler | None = None

        print(
            f"[VAD] ONNX model loaded | "
            f"threshold_high={self._threshold_high} threshold_low={self._threshold_low}",
            flush=True,
        )

    # ── Feed audio ────────────────────────────────────────────

    def update_buffer(self, audio_bytes: bytes) -> None:
        """Resample incoming audio bytes and append to internal buffer."""
        if self.resampler is None:
            self.resampler = Resampler(self.device_sample_rate, VAD_SAMPLE_RATE)
        audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        resampled = self.resampler.resample(audio_f32)
        self.buffer = np.concatenate((self.buffer, resampled))

    def update_buffer_f32(self, audio_f32: np.ndarray) -> None:
        """Accept pre-converted float32 audio (e.g. after AEC processing)."""
        if self.resampler is None:
            self.resampler = Resampler(self.device_sample_rate, VAD_SAMPLE_RATE)
        if self.device_sample_rate != VAD_SAMPLE_RATE:
            resampled = self.resampler.resample(audio_f32)
        else:
            resampled = audio_f32
        self.buffer = np.concatenate((self.buffer, resampled))

    # ── Query ─────────────────────────────────────────────────

    def get_prob(self) -> float:
        """
        Process one chunk from buffer; return latest speech probability.
        Call repeatedly until it returns 0 to drain all buffered audio.
        """
        if len(self.buffer) < MODEL_CHUNK:
            return self._latest_prob

        chunk = self.buffer[:MODEL_CHUNK].copy()
        self.buffer = self.buffer[MODEL_CHUNK:]

        # Silero v4: input=[1, chunk_size], state=[2, 1, 128]
        audio_input = chunk.reshape(1, -1).astype(np.float32)

        ort_inputs = {
            "input": audio_input,
            "state": self._h,        # reuse _h as the single state tensor
            "sr": np.array(VAD_SAMPLE_RATE, dtype=np.int64),
        }

        try:
            out, state_out = self.session.run(None, ort_inputs)
            self._h = state_out   # persist state across chunks
        except Exception as e:
            print(f"[VAD] inference error: {e}", flush=True)
            return self._latest_prob

        speech_prob = float(out.item() if hasattr(out, "item") else out[0, 0])

        # Dual-threshold hysteresis
        if speech_prob >= self._threshold_high:
            is_voice = True
        elif speech_prob <= self._threshold_low:
            is_voice = False
        else:
            is_voice = self._last_is_voice  # hold previous state
        self._last_is_voice = is_voice

        # Sliding window
        self._window.append(is_voice)
        if len(self._window) > _WINDOW_SIZE:
            self._window.pop(0)

        self._latest_prob = speech_prob
        return speech_prob

    def is_speech(self) -> bool:
        """Return True if sliding window majority is voice."""
        return self._window.count(True) >= _WINDOW_SIZE

    def empty_buffer(self) -> None:
        """Reset buffer and internal state (after wake word detection)."""
        self.buffer = np.array([], dtype=np.float32)
        self._h = np.zeros((2, 1, 128), dtype=np.float32)
        self._c = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._window = []
        self._last_is_voice = False
        self._latest_prob = 0.0
