"""
Stub: soundfile
PC-only. Stub sf.read / sf.write.
"""
import numpy as np


def read(file, dtype="float32", always_2d=False, **kw):
    """Return silence array."""
    sr = 48000
    silence = np.zeros(sr, dtype=np.float32)
    if always_2d:
        silence = silence.reshape(-1, 1)
    return silence, sr


def write(file, data, samplerate, subtype=None, **kw):
    pass


def info(file, **kw):
    class _Info:
        samplerate = 48000
        channels = 1
        frames = 48000
        format = "WAV"
        subtype = "PCM_16"
    return _Info()


class SoundFile:
    def __init__(self, file, mode="r", samplerate=None, channels=None, **kw):
        self.samplerate = samplerate or 48000
        self.channels = channels or 1
        self.frames = 0

    def read(self, frames=-1, dtype="float32", **kw):
        return np.zeros(max(frames, 0), dtype=np.float32)

    def write(self, data): pass
    def seek(self, pos, whence=0): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
