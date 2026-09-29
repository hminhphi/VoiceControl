"""
Stub: sounddevice
PC-only. No-op audio I/O cho voice_processing pipeline.
"""
import numpy as np
import threading
import time


# ── playback ────────────────────────────────────────────────────────────────

def play(data, samplerate=44100, device=None, **kw):
    pass

def wait():
    pass

def stop():
    pass

def query_devices(device=None, kind=None):
    if device is None and kind is None:
        return [{"name": "stub_device", "max_input_channels": 1, "max_output_channels": 2,
                 "default_samplerate": 48000, "index": 0}]
    return {"name": "stub_device", "max_input_channels": 1, "max_output_channels": 2,
            "default_samplerate": 48000, "index": 0, "hostapi": 0}

def query_hostapis():
    return [{"name": "stub_hostapi", "devices": [0], "default_input_device": 0, "default_output_device": 0}]

def default():
    pass


class _DefaultNS:
    device = (None, None)
    samplerate = 48000
    channels = 1


default = _DefaultNS()


# ── InputStream ─────────────────────────────────────────────────────────────

class InputStream:
    """Fake microphone — generates silence frames for pipeline compatibility."""

    def __init__(
        self,
        samplerate=48000,
        channels=1,
        dtype="float32",
        blocksize=2048,
        latency="high",
        callback=None,
        device=None,
        **kw,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback
        self._running = False
        self._thread = None

    def _run(self):
        while self._running:
            if self.callback:
                frames = np.zeros((self.blocksize, self.channels), dtype=np.float32)
                self.callback(frames, self.blocksize, {"input_buffer_adc_time": time.time()}, None)
            time.sleep(self.blocksize / self.samplerate)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False

    def close(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.close()
        return False


class OutputStream:
    def __init__(self, samplerate=48000, channels=1, dtype="float32",
                 blocksize=2048, latency="high", callback=None, device=None, **kw):
        self.samplerate = samplerate
        self.channels = channels

    def start(self): return self
    def stop(self): pass
    def close(self): pass
    def write(self, data): pass

    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── exceptions ──────────────────────────────────────────────────────────────

class PortAudioError(Exception):
    pass

class DeviceList(list):
    pass
