"""
Stub: kokoro_onnx
PC-only. Thay vì generate audio, in text ra stdout và trả về silence array.
"""
import numpy as np

SAMPLE_RATE = 24000


class Kokoro:
    def __init__(self, onnx_path, voices_path, *args, **kwargs):
        print(f"[kokoro_onnx-stub] Kokoro loaded (stub) — onnx={onnx_path}, voices={voices_path}")

    def create(self, text, voice="af_heart", speed=1.0, lang="en-us", is_phonemes=False, trim=True, **kw):
        """Return (silence_array, sample_rate) instead of real audio."""
        print(f"[TTS-STUB] Speaking: {text!r}")
        # ~0.5s of silence at 24kHz
        silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        return silence, SAMPLE_RATE

    def get_voices(self):
        return ["af_heart"]
