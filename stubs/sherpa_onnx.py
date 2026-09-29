"""
Stub: sherpa_onnx
PC-only. Fake ASR — luôn trả về empty transcript.
voice_processing/stt.py dùng khi STT_BACKEND=sherpa_onnx.
"""
import numpy as np


class _FakeToken:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    class _Text:
        text = ""
        tokens = []
    text = _Text()


class _FakeStream:
    def accept_waveform(self, sample_rate, samples): pass
    def input_finished(self): pass


class SherpaOnnx:
    pass


# ── Online / Streaming recognizer (used by stt.py) ─────────────────────────

class OnlineRecognizer:
    def __init__(self, *a, **kw):
        print("[sherpa_onnx-stub] OnlineRecognizer (stub)")

    def create_stream(self):
        return _FakeStream()

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        return _FakeResult()

    def is_endpoint(self, stream):
        return False

    def reset(self, stream):
        pass


# ── Offline recognizer ──────────────────────────────────────────────────────

class OfflineRecognizer:
    def __init__(self, *a, **kw):
        print("[sherpa_onnx-stub] OfflineRecognizer (stub)")

    def create_stream(self):
        return _FakeStream()

    def decode(self, stream):
        pass

    def get_result(self, stream):
        return _FakeResult()


# ── Keyword spotting ────────────────────────────────────────────────────────

class KeywordSpotter:
    def __init__(self, *a, **kw):
        print("[sherpa_onnx-stub] KeywordSpotter (stub)")

    def create_stream(self, keyword=""):
        return _FakeStream()

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        return _FakeResult()

    def is_endpoint(self, stream):
        return False

    def reset(self, stream):
        pass


# ── Config stubs ────────────────────────────────────────────────────────────

class OnlineRecognizerConfig:
    def __init__(self, **kw): pass

class OfflineRecognizerConfig:
    def __init__(self, **kw): pass

class OnlineSenseVoiceModelConfig:
    def __init__(self, **kw): pass

class SenseVoiceModelConfig:
    def __init__(self, **kw): pass

class OnlineTransducerModelConfig:
    def __init__(self, **kw): pass

class OnlineCtcFstDecoderConfig:
    def __init__(self, **kw): pass

class EndpointConfig:
    def __init__(self, **kw): pass

class EndpointRule:
    def __init__(self, **kw): pass

class OnlineLMConfig:
    def __init__(self, **kw): pass

class FeatureExtractorConfig:
    def __init__(self, **kw): pass
