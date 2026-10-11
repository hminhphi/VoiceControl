"""TTS via Supertonic-3 (ONNX, 44.1 kHz) — thay thế hoàn toàn Kokoro + Piper.

Ngôn ngữ: en / ja / vi (Supertonic-3 hỗ trợ 31 thứ tiếng, dự án dùng 3).
Engine chạy GPU qua onnxruntime CUDA EP; xem _install_providers().

API giữ nguyên như bản Kokoro để main.py / main_file.py / tts_to_wav.py không đổi:
  tts = TTSProcessor(sample_rate)
  tts.speech(text, language=..., output_queue=..., chunk_size=...,
             device_sample_rate=..., gen=...)
  audio, sr = tts.synthesize(text, language)     # không queue (ghi WAV)
"""
import asyncio
import os
import re
import time

import numpy as np

from lang import detect_lang

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

SUPERTONIC_SAMPLE_RATE = 44100

_HERE = os.path.dirname(os.path.abspath(__file__))
SUPERTONIC_DEFAULT_DIR = os.path.join(
    _HERE, "agent_assets", "models", "supertonic-3"
)

# language (short or locale) -> Supertonic language code
LANGUAGE_MAP = {
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "ja": "ja",
    "vi": "vi",
}

# Giọng mặc định theo ngôn ngữ (Supertonic: F1-F5 nữ, M1-M5 nam).
# Đổi qua env TTS_VOICE_EN / TTS_VOICE_JA / TTS_VOICE_VI.
DEFAULT_VOICE = "F1"
VOICE_MAP = {
    "en": "F1",
    "ja": "F1",
    "vi": "F3",
}

# Ngôn ngữ pipeline đã kiểm chứng (Supertonic-3 có nhiều hơn, nhưng chỉ 3 cái này
# được dùng và đóng model tương ứng trong bundle).
SUPPORTED_LANGS = {"en", "ja", "vi"}

# "(05:31 PM)" / "(17:30)" — LLM hay chèn lại dạng số sau khi đã viết dạng đọc;
# TTS sẽ đọc luôn phần trong ngoặc nên phải bỏ trước khi tổng hợp.
_PAREN_DIGITS_RE = re.compile(r"\s*\(([^()]*\d[^()]*)\)")

# Email / URL: Supertonic đọc thành rác ("www.fpt.com" -> "wadio.com",
# "support@fpt.com" -> "supportfbd.com" — đo 2026-10-06) nên chuyển thành
# dạng đọc trước khi tổng hợp.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
# Ngoặc đơn còn lại làm Supertonic nuốt phần đuôi câu
# ("The door (left side) is open." -> mất "is open") -> đổi thành dấu phẩy.
_PAREN_RE = re.compile(r"\(([^()]*)\)")

# "F.P.T." / "U.S." / initials "F." — fragments that must not stand alone.
_ACRONYM_RE = re.compile(r"^(?:[A-Za-z]\.)+$")


# Common TLDs stay lowercase (they are read as words, not spelled out).
_TLD_KEEP = {"com", "net", "org", "edu", "gov", "mil", "int", "io", "co"}


def _speak_domain(domain):
    """Domain -> spoken form; short non-TLD labels uppercased so acronyms
    (fpt, vn) are read as letters instead of merged into the neighbouring word."""
    labels = [
        label.upper() if 0 < len(label) <= 4 and label.lower() not in _TLD_KEEP else label
        for label in domain.split(".")
    ]
    return " dot ".join(labels)


def _email_spoken(match):
    local, _, domain = match.group(0).partition("@")
    return f"{local} at {_speak_domain(domain)}"


def _url_spoken(match):
    raw = re.sub(r"^https?://", "", match.group(0), flags=re.IGNORECASE)
    raw = re.sub(r"^www\.", "", raw, flags=re.IGNORECASE)
    domain = raw.split("/")[0].strip(".,;:!?")
    return f" {_speak_domain(domain)} " if domain else " "


def _base_lang(language):
    lang = (language or "").strip().lower()
    if not lang or lang == "auto":
        return "en"
    return lang.replace("_", "-").split("-")[0] or "en"


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _install_providers():
    """supertonic hardcode CPU (config.DEFAULT_ONNX_PROVIDERS, kem TODO cho env).
    Ghi đè trước khi tạo TTS() để session chạy trên CUDAExecutionProvider."""
    import supertonic.loader as loader

    want = (os.environ.get("ONNX_PROVIDER") or "CUDAExecutionProvider").strip()
    if want == "CPUExecutionProvider":
        providers = ["CPUExecutionProvider"]
    else:
        providers = [want, "CPUExecutionProvider"]
    loader.DEFAULT_ONNX_PROVIDERS = providers
    return providers


class TTSProcessor:
    def __init__(self, sample_rate):
        self.device_sample_rate = int(sample_rate)

        model_dir = os.environ.get("SUPERTONIC_MODEL_DIR") or SUPERTONIC_DEFAULT_DIR
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Supertonic model not found: {model_dir}\n"
                "Fetch it with: scripts/fetch_assets.py tts-supertonic"
            )

        providers = _install_providers()

        from supertonic import TTS as _SupertonicTTS

        model_name = os.environ.get("SUPERTONIC_MODEL", "supertonic-3")
        t0 = time.time()
        print(f"[TTS] Loading Supertonic {model_name}: {model_dir} providers={providers}")
        self.engine = _SupertonicTTS(
            model=model_name,
            model_dir=model_dir,
            auto_download=False,
        )
        self.engine_sr = int(self.engine.sample_rate)

        self._styles = {}
        for name in self.engine.voice_style_names:
            try:
                self._styles[name] = self.engine.get_voice_style(name)
            except Exception as e:
                print(f"[TTS] voice {name} unavailable ({e})")

        self.steps = int(os.environ.get("SUPERTONIC_STEPS", "8"))
        # 1.05 = official default: speed cao hơn làm duration predictor cắt cụt
        # từ cuối, nhất là text có số (đo 2026-10-06: vi_time 1.7 -> 1.32s cụt,
        # 1.05 -> 2.09s đủ). SUPERTONIC_SPEED_<LANG> vẫn override được.
        self.speed = float(os.environ.get("SUPERTONIC_SPEED", "1.05"))

        print(
            f"[TTS] Supertonic loaded OK: sr={self.engine_sr} "
            f"voices={sorted(self._styles)} steps={self.steps} speed={self.speed} "
            f"({time.time() - t0:.2f}s)"
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _map_lang(self, language):
        return LANGUAGE_MAP.get(_base_lang(language), "en")

    def _voice_for(self, language):
        base = _base_lang(language)
        voice = os.environ.get(f"TTS_VOICE_{base.upper()}") or VOICE_MAP.get(base, DEFAULT_VOICE)
        if voice not in self._styles:
            print(f"[TTS] voice {voice!r} not in model; using {DEFAULT_VOICE}")
            voice = DEFAULT_VOICE if DEFAULT_VOICE in self._styles else sorted(self._styles)[0]
        return voice

    def _speed_for(self, language):
        """Per-language speed override: SUPERTONIC_SPEED_VI / _JA / _EN.

        Vietnamese measures ~40% slower per word than English at the same
        speed (2026-10-06: 2.6 vs 3.7 words/s at speed=1.2); speed=1.7
        matches the English pace (3.7 words/s).
        """
        base = _base_lang(language)
        raw = os.environ.get(f"SUPERTONIC_SPEED_{base.upper()}")
        if raw:
            try:
                return float(raw)
            except ValueError:
                print(f"[TTS] invalid SUPERTONIC_SPEED_{base.upper()}={raw!r}; using {self.speed}")
        return self.speed

    def _trim_silence(self, audio, sr):
        """Trim leading/trailing near-silence, keeping a small margin.

        Supertonic pads ~0.4-0.8s of silence per utterance (VI pads more than
        EN), which makes replies feel slow. Disable with TTS_TRIM_SILENCE=0.
        """
        if not _env_flag("TTS_TRIM_SILENCE", True) or audio is None or len(audio) == 0:
            return audio
        threshold = float(os.environ.get("TTS_TRIM_THRESHOLD", "0.01"))
        margin_ms = float(os.environ.get("TTS_TRIM_MARGIN_MS", "50"))
        idx = np.nonzero(np.abs(audio) > threshold)[0]
        if len(idx) == 0:
            return audio
        margin = int(margin_ms / 1000.0 * sr)
        start = max(0, int(idx[0]) - margin)
        end = min(len(audio), int(idx[-1]) + margin)
        return audio[start:end]

    def _sanitize(self, text):
        """Chuẩn hóa text trước khi tổng hợp (các lỗi đo được 2026-10-06).

        - "(05:31 PM)" (ngoặc chứa số): bỏ hẳn — TTS sẽ đọc cả phần trong ngoặc.
        - "(...)" còn lại -> ", ... ," — ngoặc làm Supertonic nuốt đuôi câu.
        - Email/URL -> dạng đọc ("support at fpt dot com") — đọc thẳng thành rác.
        """
        text = text or ""
        text = _PAREN_DIGITS_RE.sub("", text)
        text = _EMAIL_RE.sub(_email_spoken, text)
        text = _URL_RE.sub(_url_spoken, text)
        text = _PAREN_RE.sub(
            lambda m: f", {m.group(1).strip()}, " if m.group(1).strip() else " ",
            text,
        )
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*,\s*(?:,\s*)+", ", ", text)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text.strip(" ,")

    def _emit(self, audio, sr, output_queue, chunk_size, device_sample_rate, gen):
        if output_queue is not None and chunk_size is not None and device_sample_rate is not None:
            from resampler import Resampler
            if sr != device_sample_rate:
                audio = Resampler(sr, device_sample_rate).resample(audio)
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=0)
                elif len(chunk) > chunk_size:
                    chunk = chunk[:chunk_size]
                item = chunk.reshape(chunk_size, 1).astype(np.float32)
                output_queue.put((gen, item) if gen is not None else item)
        else:
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            if HAS_SOUNDDEVICE:
                sd.play(audio_int16, sr)
                sd.wait()
            else:
                print("[TTS] sounddevice not available, skip playback")

    def _emit_chunk(self, audio_chunk, sr, output_queue, chunk_size, device_sample_rate, gen, resampler=None):
        """Emit a single audio chunk to the output queue (for streaming)."""
        if output_queue is not None and chunk_size is not None and device_sample_rate is not None:
            if resampler is None and sr != device_sample_rate:
                from resampler import Resampler
                resampler = Resampler(sr, device_sample_rate)
            if resampler is not None:
                audio_chunk = resampler.resample(audio_chunk)
            # Split into device-sized chunks
            for i in range(0, len(audio_chunk), chunk_size):
                chunk = audio_chunk[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=0)
                elif len(chunk) > chunk_size:
                    chunk = chunk[:chunk_size]
                item = chunk.reshape(chunk_size, 1).astype(np.float32)
                output_queue.put((gen, item) if gen is not None else item)
        else:
            self._emit(audio_chunk, sr, output_queue, chunk_size, device_sample_rate, gen)

    def _split_sentences(self, text):
        """Split text into sentences for streaming TTS.

        Latin punctuation only ends a sentence when followed by whitespace, so
        acronyms/decimals stay intact ("F.P.T. Automotive", "3.5 km"); CJK
        punctuation splits directly. Acronym fragments that still end up alone
        ("F.P.T." / "U.S." / initials "F.") are merged into the next sentence:
        a lone fragment loses its context and Supertonic mangles it (heard as
        "F." or dropped entirely, 2026-10-06).
        """
        sentences = re.split(r"(?<=[。！？])\s*|(?<=[.!?])\s+", text)
        parts = [s.strip() for s in sentences if s.strip()]
        merged = []
        for part in parts:
            if merged and _ACRONYM_RE.match(merged[-1]):
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        return merged

    # ── core synthesis ─────────────────────────────────────────────────────
    def synthesize(self, text, language=None):
        """Return (audio_float32 mono, sample_rate) — engine mới, dùng cho
        tts_to_wav.py và các test."""
        text = self._sanitize(text)
        lang = self._map_lang(language)
        voice = self._voice_for(language)
        wav, _info = self.engine.synthesize(
            text,
            self._styles[voice],
            lang=lang,
            total_steps=self.steps,
            speed=self._speed_for(language),
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio = self._trim_silence(audio, self.engine_sr)
        return audio, self.engine_sr

    async def _synthesize_stream_async(self, text, language, output_queue, chunk_size, device_sample_rate, gen):
        """Streaming theo từng câu (Supertonic không có API stream nội bộ)."""
        text = self._sanitize(text)
        voice = self._voice_for(language)
        lang = self._map_lang(language)

        sentences = self._split_sentences(text) or [text]
        print(f"[TTS] Streaming {len(sentences)} sentences: lang={lang} voice={voice}")

        resampler = None
        for idx, sentence in enumerate(sentences, 1):
            print(f"[TTS] Sentence {idx}/{len(sentences)}: {sentence!r}")
            wav, _info = self.engine.synthesize(
                sentence,
                self._styles[voice],
                lang=lang,
                total_steps=self.steps,
                speed=self._speed_for(language),
            )
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            audio = self._trim_silence(audio, self.engine_sr)
            resampler = self._emit_chunk(
                audio, self.engine_sr, output_queue, chunk_size,
                device_sample_rate, gen, resampler,
            )
        print(f"[TTS] Streamed {len(sentences)} sentences")

    def _synthesize_stream(self, text, language, output_queue, chunk_size, device_sample_rate, gen):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._synthesize_stream_async(
                    text, language, output_queue, chunk_size, device_sample_rate, gen
                )
            )
        finally:
            loop.close()

    def speech(self, text, language=None, output_queue=None, chunk_size=None, device_sample_rate=None, gen=None):
        # gen: barge-in generation tag. When set, each audio chunk is queued
        # as (gen, chunk) so playback can drop audio from aborted turns.
        if not text or not text.strip():
            return
        base = _base_lang(language)
        if base not in SUPPORTED_LANGS:
            # Last-resort: recover the language from the reply text itself
            # (e.g. an STT backend reported an unsupported code like "nn").
            detected = detect_lang(text)
            if detected:
                print(f"[TTS] language {language!r} not supported; using detected {detected!r}")
                language = detected
            else:
                print(f"[TTS] language {language!r} not supported; falling back to English")
        t0 = time.time()
        try:
            # Use streaming when output_queue is provided (real-time playback)
            if output_queue is not None and chunk_size is not None and device_sample_rate is not None:
                self._synthesize_stream(text.strip(), language, output_queue, chunk_size, device_sample_rate, gen)
            else:
                # Non-streaming fallback (no queue - direct playback)
                audio, sr = self.synthesize(text.strip(), language)
                self._emit(audio, sr, output_queue, chunk_size, device_sample_rate, gen)
        except Exception as e:
            print(f"[TTS] Error: {e}")
            import traceback
            traceback.print_exc()
        print(f"[TTS] Done ({time.time() - t0:.2f}s)")


# Tên cũ giữ để tương thích (tts_to_wav.py import KOKORO_SAMPLE_RATE).
KOKORO_SAMPLE_RATE = SUPERTONIC_SAMPLE_RATE
