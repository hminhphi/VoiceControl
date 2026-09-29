"""
Stub: openwakeword
PC-only. Model stub không bao giờ detect wake word.
Pipeline trong main.py sẽ tiếp tục loop mà không trigger.
"""
import numpy as np


class Model:
    def __init__(self, *args, wakeword_models=None, inference_framework="onnx", **kwargs):
        self.wakeword_models = wakeword_models or []
        print(f"[openwakeword-stub] Model loaded (stub) — models={wakeword_models}")
        # Prediction dict dùng để check trong wake_word.py
        self.prediction_buffer = {}
        for m in self.wakeword_models:
            name = m.replace("/", "_").replace("\\", "_").replace(".onnx", "")
            self.prediction_buffer[name] = [0.0] * 16

    def predict(self, audio_chunk, **kw):
        """Always return scores below threshold."""
        result = {}
        for key in self.prediction_buffer:
            result[key] = 0.0
        return result

    def predict_clip(self, audio_data, **kw):
        return self.predict(audio_data)

    @property
    def models(self):
        return {}
