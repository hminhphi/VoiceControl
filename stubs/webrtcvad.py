"""
Stub: webrtcvad
PC-only. Luôn trả về False (no speech detected).
"""

class Vad:
    def __init__(self, mode=3):
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def is_speech(self, buf, sample_rate, length=None):
        return False


# Module-level alias used by some imports
class WebRtcVad(Vad):
    pass
