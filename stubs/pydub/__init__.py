"""
Stub: pydub
PC-only. Minimal AudioSegment stub cho stt.py.
"""
import numpy as np
import io


class AudioSegment:
    def __init__(self, data=b"", sample_width=2, frame_rate=16000, channels=1):
        self._data = data
        self.sample_width = sample_width
        self.frame_rate = frame_rate
        self.channels = channels
        self.frame_width = sample_width * channels

    @classmethod
    def from_wav(cls, file):
        return cls()

    @classmethod
    def from_file(cls, file, format=None, **kw):
        return cls()

    @classmethod
    def from_raw(cls, data, sample_width=2, frame_rate=16000, channels=1):
        return cls(data, sample_width, frame_rate, channels)

    @classmethod
    def silent(cls, duration=0, frame_rate=16000):
        n = int(duration / 1000 * frame_rate * 2)
        return cls(b"\x00" * n, frame_rate=frame_rate)

    def export(self, out_f=None, format="wav", **kw):
        if out_f is None:
            return io.BytesIO(self._data)
        if hasattr(out_f, "write"):
            out_f.write(self._data)
        return out_f

    def get_array_of_samples(self):
        return np.frombuffer(self._data or b"\x00\x00", dtype=np.int16)

    def set_frame_rate(self, rate):
        return AudioSegment(self._data, self.sample_width, rate, self.channels)

    def set_channels(self, channels):
        return AudioSegment(self._data, self.sample_width, self.frame_rate, channels)

    def __add__(self, other):
        return AudioSegment(
            self._data + other._data,
            self.sample_width,
            self.frame_rate,
            self.channels,
        )

    def __len__(self):
        if not self._data or self.frame_width == 0 or self.frame_rate == 0:
            return 0
        return int(len(self._data) / self.frame_width / self.frame_rate * 1000)

    def __getitem__(self, sl):
        return AudioSegment(b"", self.sample_width, self.frame_rate, self.channels)

    @property
    def raw_data(self):
        return self._data

    @property
    def duration_seconds(self):
        return len(self) / 1000.0
