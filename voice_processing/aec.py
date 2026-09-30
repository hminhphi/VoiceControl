"""
AEC (Acoustic Echo Cancellation) Engine cho voice_processing.

Ported từ py-xiaozhi AecEngine, được adapt cho pipeline hiện tại:
- near: audio từ microphone (device sample rate, int16/float32)
- far: audio đang phát ra loa (output_queue playback chunks)

Flow:
    Mic callback  →  feed_far(outdata) [output thread, fast]
                  →  process_near(indata) [capture thread]
                       ├── drain far queue → resample to 16kHz → ProcessReverseStream
                       └── ProcessStream → echo-cancelled output

Graceful degradation:
    - Nếu webrtc_apm không available → active=False, passthrough
    - Nếu soxr không available → dùng numpy linear interp

Environment variables:
    AEC_ENABLED=1          (default: 1, set 0 to disable)
    AEC_DELAY_MS=60        playback-to-capture delay estimate
    AEC_NOISE_SUPPRESS=1   enable NS + high-pass filter
    AEC_AGC_ENABLED=1      enable AGC2 adaptive digital leveling (default: 1)
    AEC_NS_LEVEL=high      NS strength: low|moderate|high|very_high (default: high)
    AEC_TRANSIENT_SUPPRESS=1 transient click suppression (default: 1)
    AEC_NS_LINEAR=0        NS analyses linear AEC output (stronger; default: 0)
    AEC_HPF_FULL_BAND=1    high-pass applies in full band (default: 1)
    AEC_MOBILE_MODE=0      echo canceller mobile mode (default: 0)
    AEC_EXPORT_LINEAR=0    export linear AEC output (default: 0)
    AEC_AGC1_ENABLED=0     optional AGC1 target-level loudness + limiter (default: 0)
    AEC_AGC1_TARGET_DBFS=-3  AGC1 target level (default: -3)
    AEC_AGC1_COMPRESSION_DB=9 AGC1 compression gain (default: 9)
    AEC_AGC1_LIMITER=1     AGC1 limiter (default: 1)
    AEC_AGC_HEADROOM_DB=5  target headroom below clipping (default: 5.0)
    AEC_AGC_MAX_GAIN_DB=30 max adaptive boost for weak mics (default: 30.0)
    AEC_AGC_INITIAL_GAIN_DB=15 starting gain, converges down/up (default: 15.0)
    AEC_AGC_MAX_SPEED_DB=12 max gain slew rate dB/s, faster onset after silence (default: 12.0)
    AEC_AGC_MAX_NOISE_DBFS=-50.0 noise floor cap so silence isn't pumped (default)
    AEC_PRE_GAIN=1.0       fixed linear pre-amp factor, 1.0 = off (default)
"""

import ctypes
import os
import threading
from collections import deque

import numpy as np

_MAX_CONSECUTIVE_FAILURES = 5
_FAR_PENDING_MAX_BLOCKS = 25


def _env_bool(name: str, default: bool = True) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[AEC][config] invalid {name}={raw!r}; using {default}", flush=True)
        return default


class AecEngine:
    """
    WebRTC APM wrapper: AEC + optional NS + high-pass filter.

    Usage::
        aec = AecEngine(near_rate=48000, far_rate=48000)
        # In output callback (fast path):
        aec.feed_far(outdata)
        # In capture thread (slow path, OK to block briefly):
        processed = aec.process_near(indata_float32)
    """

    def __init__(
        self,
        near_rate: int = 48000,
        far_rate: int = 48000,
        delay_ms: int = 60,
        enable_preprocess: bool = True,
    ):
        self._near_rate = int(near_rate)
        self._far_rate = int(far_rate)
        self._delay_ms = int(delay_ms)
        # WebRTC APM processes in 10 ms frames at 16 kHz → 160 samples
        self._apm_rate = 16000
        self._frame = self._apm_rate // 100  # 160 samples

        self._lock = threading.Lock()
        self._active = False
        self._closed = False
        self._fail_count = 0

        self._apm = None
        self._stream_cfg = None

        self._far_pending: deque = deque()
        self._far_buffer = np.empty(0, dtype=np.float32)
        self._near_buffer = np.empty(0, dtype=np.float32)
        self._near_resampler = None   # device_rate → 16kHz
        self._far_resampler = None    # device_rate → 16kHz
        self._out_resampler = None    # 16kHz → device_rate
        self._far_dropped = 0

        # ctypes buffers (reused to avoid alloc in hot path)
        self._near_in = (ctypes.c_short * self._frame)()
        self._near_out = (ctypes.c_short * self._frame)()
        self._far_in = (ctypes.c_short * self._frame)()
        self._far_out = (ctypes.c_short * self._frame)()

        if not _env_bool("AEC_ENABLED", True):
            print("[AEC] Disabled via AEC_ENABLED=0", flush=True)
            return

        try:
            self._init_apm(enable_preprocess)
            self._init_resamplers()
            self._active = True
            print(
                f"[AEC] Active | near={self._near_rate}Hz far={self._far_rate}Hz "
                f"apm={self._apm_rate}Hz delay={self._delay_ms}ms "
                f"preprocess={enable_preprocess} "
                f"agc={getattr(self, '_agc_enabled', False)} "
                f"agc1={getattr(self, '_agc1_enabled', False)} "
                f"ns={getattr(self, '_ns_level', 'n/a')} "
                f"ns_linear={getattr(self, '_ns_linear', False)} "
                f"pre_gain={getattr(self, '_pre_gain', 1.0)}",
                flush=True,
            )
        except Exception as e:
            print(f"[AEC] Init failed — passthrough mode: {e}", flush=True)
            self._release()

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_apm(self, enable_preprocess: bool) -> None:
        try:
            from libs import webrtc_apm as apm_mod
        except ImportError:
            raise RuntimeError(
                "webrtc_apm not found. Install py-xiaozhi's libs/webrtc_apm "
                "or set AEC_ENABLED=0 to disable."
            )

        self._apm = apm_mod.WebRTCAudioProcessing()
        config = apm_mod.create_default_config()
        config.echo.enabled = True
        config.echo.mobile_mode = _env_bool("AEC_MOBILE_MODE", False)
        config.echo.export_linear_aec_output = _env_bool("AEC_EXPORT_LINEAR", False)
        if enable_preprocess:
            config.high_pass.enabled = True
            config.high_pass.apply_in_full_band = _env_bool("AEC_HPF_FULL_BAND", True)
            config.noise_suppress.enabled = True
            # NS can analyse the linear AEC output for stronger suppression.
            config.noise_suppress.analyze_linear_aec_output_when_available = _env_bool(
                "AEC_NS_LINEAR", False
            )
            # Env: AEC_NS_LEVEL=low|moderate|high|very_high (default: high,
            # xiaozhi-style stronger suppression for noisy rooms).
            ns_level = os.environ.get("AEC_NS_LEVEL", "high").strip().lower()
            ns_map = {
                "low": apm_mod.NoiseSuppressionLevel.LOW,
                "moderate": apm_mod.NoiseSuppressionLevel.MODERATE,
                "high": apm_mod.NoiseSuppressionLevel.HIGH,
                "very_high": apm_mod.NoiseSuppressionLevel.VERY_HIGH,
            }
            if ns_level not in ns_map:
                print(f"[AEC][config] invalid AEC_NS_LEVEL={ns_level!r}; using high", flush=True)
                ns_level = "high"
            config.noise_suppress.noise_level = ns_map[ns_level]
            self._ns_level = ns_level
            # Transient suppressor: keyboard clicks, taps (xiaozhi AFE-style).
            # Env: AEC_TRANSIENT_SUPPRESS=1 (default: 1).
            config.transient_suppress.enabled = _env_bool("AEC_TRANSIENT_SUPPRESS", True)

        # ── AGC2 adaptive digital (xiaozhi-style leveling) ──────────
        # Pure-digital gain: needs no OS mic-volume control, so it works
        # the same on host (Windows) and in Docker (Linux). Lifts weak
        # mics (e.g. -45 dBFS RMS) up toward headroom without pumping
        # background noise past max_output_noise_level_dbfs.
        # Env: AEC_AGC_ENABLED=1, AEC_AGC_MAX_GAIN_DB=30,
        #      AEC_AGC_INITIAL_GAIN_DB=15, AEC_AGC_HEADROOM_DB=5
        self._agc_enabled = _env_bool("AEC_AGC_ENABLED", True)
        if self._agc_enabled:
            gc2 = config.gain_control2
            gc2.enabled = True
            gc2.adaptive_controller.enabled = True
            gc2.adaptive_controller.headroom_db = _env_float("AEC_AGC_HEADROOM_DB", 5.0)
            gc2.adaptive_controller.max_gain_db = _env_float("AEC_AGC_MAX_GAIN_DB", 30.0)
            gc2.adaptive_controller.initial_gain_db = _env_float("AEC_AGC_INITIAL_GAIN_DB", 15.0)
            gc2.adaptive_controller.max_gain_change_db_per_second = _env_float(
                "AEC_AGC_MAX_SPEED_DB", 12.0
            )
            gc2.adaptive_controller.max_output_noise_level_dbfs = _env_float(
                "AEC_AGC_MAX_NOISE_DBFS", -50.0
            )

        # ── Optional AGC1 (target-level loudness + limiter) ──────────
        # Complements AGC2: drives the level toward a fixed target and limits
        # peaks. Useful to make a weak mic loud enough for the ASR model.
        # Env: AEC_AGC1_ENABLED=1, AEC_AGC1_TARGET_DBFS=-3,
        #      AEC_AGC1_COMPRESSION_DB=9, AEC_AGC1_LIMITER=1
        self._agc1_enabled = _env_bool("AEC_AGC1_ENABLED", False)
        self._ns_linear = _env_bool("AEC_NS_LINEAR", False)
        if self._agc1_enabled:
            gc1 = config.gain_control1
            gc1.enabled = True
            gc1.controller_mode = apm_mod.GainController1Mode.ADAPTIVE_DIGITAL
            gc1.target_level_dbfs = int(_env_float("AEC_AGC1_TARGET_DBFS", -3.0))
            gc1.compression_gain_db = int(_env_float("AEC_AGC1_COMPRESSION_DB", 9.0))
            gc1.enable_limiter = _env_bool("AEC_AGC1_LIMITER", True)

        # ── Optional fixed pre-amp (manual override, default off) ───
        # Env: AEC_PRE_GAIN=2.0  (linear factor; 1.0 = disabled)
        self._pre_gain = _env_float("AEC_PRE_GAIN", 1.0)
        if self._pre_gain != 1.0:
            config.pre_amp.enabled = True
            config.pre_amp.fixed_gain_factor = float(self._pre_gain)

        ret = self._apm.apply_config(config)
        if ret != 0:
            raise RuntimeError(f"apply_config returned {ret}")

        self._stream_cfg = self._apm.create_stream_config(self._apm_rate, 1)
        self._apm.set_stream_delay_ms(self._delay_ms)

    def _init_resamplers(self) -> None:
        """Try soxr for quality; fall back to numpy linear interpolation."""
        if self._near_rate == self._apm_rate and self._far_rate == self._apm_rate:
            return  # no resampling needed
        try:
            import soxr

            if self._near_rate != self._apm_rate:
                self._near_resampler = soxr.ResampleStream(
                    self._near_rate, self._apm_rate, 1, dtype="float32", quality="QQ"
                )
                self._out_resampler = soxr.ResampleStream(
                    self._apm_rate, self._near_rate, 1, dtype="float32", quality="QQ"
                )
            if self._far_rate != self._apm_rate:
                self._far_resampler = soxr.ResampleStream(
                    self._far_rate, self._apm_rate, 1, dtype="float32", quality="QQ"
                )
        except ImportError:
            print("[AEC] soxr not available, using numpy resampler", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._active

    def feed_far(self, outdata: np.ndarray) -> None:
        """
        Output callback thread: enqueue far-end (speaker) audio.
        Fast path — only downmix + copy, no resampling.
        """
        if not self._active:
            return
        try:
            if outdata.ndim > 1 and outdata.shape[1] > 1:
                mono = outdata.mean(axis=1).astype(np.float32)
            else:
                mono = np.asarray(outdata, dtype=np.float32).ravel()

            self._far_pending.append(mono.copy())
            while len(self._far_pending) > _FAR_PENDING_MAX_BLOCKS:
                self._far_pending.popleft()
                self._far_dropped += 1
        except Exception:
            pass  # output path must never raise

    def process_near(self, block: np.ndarray) -> np.ndarray:
        """
        Capture thread: apply AEC to microphone audio.

        Args:
            block: float32 mono array at device sample rate.
        Returns:
            Echo-cancelled float32 mono array at the SAME device sample rate.
            Returns the original block on error (graceful passthrough).
        """
        if not self._active:
            return block

        try:
            # Step 1: resample near to APM rate
            near_16k = self._resample_near(block)

            # Step 2: resample + drain far queue into APM
            with self._lock:
                if not self._active:
                    return block
                self._drain_far_locked()
                self._apm.set_stream_delay_ms(self._delay_ms)

                # Step 3: process near in 10 ms frames
                out_16k = self._run_near_frames(near_16k)

            # Step 4: resample output back to device rate
            result = self._resample_output(out_16k, len(block))
            self._fail_count = 0
            return result

        except Exception as e:
            self._on_failure("near", e)
            return block

    def set_delay_ms(self, delay_ms: int) -> None:
        self._delay_ms = int(delay_ms)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._active = False
            self._release()
        print("[AEC] Closed", flush=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resample_near(self, block: np.ndarray) -> np.ndarray:
        if self._near_rate == self._apm_rate:
            return block.astype(np.float32)
        if self._near_resampler is not None:
            return self._near_resampler.resample_chunk(block.astype(np.float32))
        return self._numpy_resample(block.astype(np.float32), self._near_rate, self._apm_rate)

    def _resample_output(self, out_16k: np.ndarray, target_len: int) -> np.ndarray:
        if self._near_rate == self._apm_rate:
            return out_16k
        if self._out_resampler is not None:
            return self._out_resampler.resample_chunk(out_16k)
        return self._numpy_resample(out_16k, self._apm_rate, self._near_rate)

    def _run_near_frames(self, near_16k: np.ndarray) -> np.ndarray:
        n = len(near_16k)
        # Pad to frame boundary
        remainder = n % self._frame
        if remainder:
            near_16k = np.concatenate(
                [near_16k, np.zeros(self._frame - remainder, dtype=np.float32)]
            )
        out = np.empty(len(near_16k), dtype=np.float32)
        i16 = self._f32_to_i16(near_16k)
        for off in range(0, len(near_16k), self._frame):
            ctypes.memmove(self._near_in, i16[off: off + self._frame].ctypes.data, self._frame * 2)
            ret = self._apm.process_stream(self._near_in, self._stream_cfg, self._stream_cfg, self._near_out)
            if ret != 0:
                raise RuntimeError(f"process_stream returned {ret}")
            out[off: off + self._frame] = (
                np.frombuffer(self._near_out, dtype=np.int16).astype(np.float32) / 32768.0
            )
        return out[:n]  # trim back to original length

    def _drain_far_locked(self) -> None:
        """Resample and feed all pending far-end frames to APM (call under lock)."""
        while self._far_pending:
            mono = self._far_pending.popleft()
            if self._far_resampler is not None:
                mono = self._far_resampler.resample_chunk(mono)
            elif self._far_rate != self._apm_rate:
                mono = self._numpy_resample(mono, self._far_rate, self._apm_rate)
            if len(mono):
                self._far_buffer = np.concatenate((self._far_buffer, mono))

        n_frames = len(self._far_buffer) // self._frame
        if n_frames == 0:
            return

        usable = n_frames * self._frame
        i16 = self._f32_to_i16(self._far_buffer[:usable])
        self._far_buffer = self._far_buffer[usable:]

        for off in range(0, usable, self._frame):
            ctypes.memmove(self._far_in, i16[off: off + self._frame].ctypes.data, self._frame * 2)
            ret = self._apm.process_reverse_stream(
                self._far_in, self._stream_cfg, self._stream_cfg, self._far_out
            )
            if ret != 0:
                raise RuntimeError(f"process_reverse_stream returned {ret}")

    def _release(self) -> None:
        self._active = False
        try:
            if self._apm is not None and self._stream_cfg is not None:
                self._apm.destroy_stream_config(self._stream_cfg)
        except Exception:
            pass
        self._stream_cfg = None
        self._apm = None
        self._near_resampler = None
        self._far_resampler = None
        self._out_resampler = None
        self._far_pending.clear()
        self._far_buffer = np.empty(0, dtype=np.float32)
        if self._far_dropped:
            print(f"[AEC] far dropped total: {self._far_dropped}", flush=True)

    def _on_failure(self, side: str, err: Exception) -> None:
        self._fail_count += 1
        if self._fail_count >= _MAX_CONSECUTIVE_FAILURES:
            print(
                f"[AEC] {side} failed {self._fail_count}x — bypassing: {err}",
                flush=True,
            )
            with self._lock:
                self._release()
        else:
            print(f"[AEC] {side} failure #{self._fail_count}: {err}", flush=True)

    @staticmethod
    def _f32_to_i16(x: np.ndarray) -> np.ndarray:
        return np.clip(x * 32768.0, -32768.0, 32767.0).astype(np.int16)

    @staticmethod
    def _numpy_resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return x
        n_out = int(len(x) * dst_rate / src_rate)
        return np.interp(
            np.linspace(0, len(x) - 1, n_out),
            np.arange(len(x)),
            x,
        ).astype(np.float32)
