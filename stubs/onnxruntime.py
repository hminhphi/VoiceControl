"""
Stub: onnxruntime
PC-only. Fake ONNX inference sessions cho VAD.
Trên Jetson onnxruntime-gpu thật được dùng.
"""
import numpy as np


class SessionOptions:
    def __init__(self):
        self.inter_op_num_threads = 1
        self.intra_op_num_threads = 1
        self.graph_optimization_level = 99


class InferenceSession:
    def __init__(self, model_path, providers=None, sess_options=None, **kw):
        self._model_path = model_path
        print(f"[onnxruntime-stub] InferenceSession({model_path}) — returning zeros")
        # Detect VAD model by heuristic
        self._is_vad = "silero" in str(model_path).lower() or "vad" in str(model_path).lower()

    def run(self, output_names, input_feed, **kw):
        """Return plausible dummy output matching Silero VAD v4/v5 signature."""
        if self._is_vad:
            # Silero VAD: output[0] is speech_prob [1,1], output[1] is state
            prob = np.array([[0.01]], dtype=np.float32)  # Very low = silence
            state = np.zeros((2, 1, 128), dtype=np.float32)
            return [prob, state]
        # Generic: return zeros
        return [np.zeros(1, dtype=np.float32)]

    def get_inputs(self):
        class _Meta:
            name = "input"
            shape = [1, 512]
            type = "tensor(float)"
        return [_Meta()]

    def get_outputs(self):
        class _Meta:
            name = "output"
            shape = [1, 1]
            type = "tensor(float)"
        return [_Meta()]


# ── Providers ────────────────────────────────────────────────────────────────

def get_available_providers():
    return ["CPUExecutionProvider"]

def get_device():
    return "CPU"


class GraphOptimizationLevel:
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_ALL = 99
