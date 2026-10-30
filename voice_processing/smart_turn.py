"""Semantic end-of-turn detection — Smart Turn v3.2 (ONNX, CPU).

Predicts whether the last 8 s of audio form a *finished* turn using grammar /
prosody, which a fixed silence timer cannot. The pipeline uses it to veto a
premature turn end (user paused mid-sentence -> keep listening).

Model: pipecat-ai/smart-turn-v3 (`smart-turn-v3.2-cpu.onnx`, 8 MB, BSD-2).
Input: Whisper log-mel [1, 80, 800] (8 s @ 16 kHz, zero-mean/unit-var norm).
Output: sigmoid probability that the turn is complete.

Env:
  SMART_TURN_ENABLED    1 = on (default), 0 = off
  SMART_TURN_THRESHOLD  prob below this = "still speaking" (default 0.35)
  SMART_TURN_MODEL      optional model path override
"""
import os

import numpy as np
import onnxruntime as ort

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
CHUNK_SEC = 8
N_FRAMES = CHUNK_SEC * 100
MIN_AUDIO_SEC = 0.5

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(
    _HERE, "agent_assets", "models", "smart-turn", "smart-turn-v3.2-cpu.onnx"
)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    linear = f / (200.0 / 3.0)
    return np.where(
        f >= min_log_hz,
        min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep,
        linear,
    )


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    linear = (200.0 / 3.0) * m
    return np.where(
        m >= min_log_mel,
        min_log_hz * np.exp(logstep * (m - min_log_mel)),
        linear,
    )


def _mel_filterbank():
    """80-bin Slaney mel filterbank (identical to WhisperFeatureExtractor)."""
    num_bins = 1 + N_FFT // 2
    fft_freqs = np.linspace(0, SR // 2, num_bins)
    mels = np.linspace(_hz_to_mel(0.0), _hz_to_mel(8000.0), N_MELS + 2)
    hz = _mel_to_hz(mels)
    fb = np.zeros((N_MELS, num_bins), dtype=np.float64)
    for i in range(N_MELS):
        left, center, right = hz[i], hz[i + 1], hz[i + 2]
        rising = (fft_freqs - left) / max(center - left, 1e-12)
        falling = (right - fft_freqs) / max(right - center, 1e-12)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
        fb[i] *= 2.0 / max(right - left, 1e-12)
    return fb


class SmartTurnDetector:
    def __init__(self, model_path=None):
        path = model_path or os.environ.get("SMART_TURN_MODEL") or DEFAULT_MODEL
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"smart-turn model not found: {path}\n"
                "Fetch it with: scripts/fetch_assets.py smart-turn"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.fb = _mel_filterbank()
        n = np.arange(N_FFT)
        self.window = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / N_FFT))

    def _log_mel(self, audio):
        audio = np.asarray(audio, dtype=np.float64)
        if len(audio) > CHUNK_SEC * SR:
            audio = audio[-CHUNK_SEC * SR:]
        valid = len(audio)
        if valid < CHUNK_SEC * SR:
            audio = np.pad(audio, (0, CHUNK_SEC * SR - valid))
        if valid > 0:
            mean = audio[:valid].mean()
            var = audio[:valid].var()
            audio = audio.copy()
            audio[:valid] = (audio[:valid] - mean) / np.sqrt(var + 1e-7)
        padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
        n_frames = 1 + (len(padded) - N_FFT) // HOP
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
        frames = padded[idx] * self.window
        spec = np.fft.rfft(frames, n=N_FFT, axis=1)
        power = (np.abs(spec) ** 2).T
        mel = self.fb @ power
        log_spec = np.log10(np.maximum(mel, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if log_spec.shape[1] < N_FRAMES:
            log_spec = np.pad(log_spec, ((0, 0), (0, N_FRAMES - log_spec.shape[1])))
        return log_spec[:, :N_FRAMES]

    def predict(self, audio):
        feats = self._log_mel(audio)[None, :, :].astype(np.float32)
        out = self.sess.run(None, {"input_features": feats})[0]
        value = float(np.ravel(out)[0])
        prob = value if 0.0 <= value <= 1.0 else 1.0 / (1.0 + np.exp(-value))
        return {"prediction": 1 if prob > 0.5 else 0, "probability": prob}


def detect_turn_complete(audio):
    """Return probability that `audio` (float32 16 kHz, <=8 s) ends a turn."""
    if len(audio) < MIN_AUDIO_SEC * SR:
        return None
    det = _get_detector()
    if det is None:
        return None
    return det.predict(audio)["probability"]


_DETECTOR = None
_DETECTOR_FAILED = False


def _get_detector():
    global _DETECTOR, _DETECTOR_FAILED
    if _DETECTOR is not None or _DETECTOR_FAILED:
        return _DETECTOR
    try:
        _DETECTOR = SmartTurnDetector()
        print("[SMART-TURN] loaded (semantic end-of-turn, CPU)", flush=True)
    except Exception as e:
        _DETECTOR_FAILED = True
        print(f"[SMART-TURN] unavailable: {e}", flush=True)
    return _DETECTOR
