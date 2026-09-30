import glob
import os
import time
import numpy as np
from kokoro_onnx import Kokoro

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

KOKORO_SAMPLE_RATE = 24000

_HERE = os.path.dirname(os.path.abspath(__file__))
PIPER_VI_DEFAULT_DIR = os.path.join(
    _HERE, "agent_assets", "models", "tts", "vits-piper-vi_VN-vais1000-medium"
)

# language (short or locale) -> Kokoro language code understood by the tokenizer
LANGUAGE_MAP = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "ja": "ja",
    "zh": "zh",
    "cmn": "zh",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "pt": "pt",
    "hi": "hi",
}

# Fallback voice per language. Kokoro v1.0 ships English + Japanese (+ zh/es/fr/...).
# Japanese uses a dedicated voice; English keeps the original default voice.
DEFAULT_VOICE = "af_heart"
VOICE_MAP = {
    "en": "af_heart",
    "ja": "jf_alpha",
}
# Languages the pipeline is validated to speak. English/Japanese via Kokoro,
# Vietnamese via sherpa-onnx Piper. Others fall back to English.
SUPPORTED_LANGS = {"en", "ja", "vi"}

# misaki[ja] G2P is required for Kokoro's Japanese voices (the bundled
# tokenizer only does espeak-ng, which does not match Kokoro's ja phoneme set).
_misaki_ja = None
_misaki_ja_checked = False


def _get_misaki_ja():
    global _misaki_ja, _misaki_ja_checked
    if not _misaki_ja_checked:
        _misaki_ja_checked = True
        try:
            from misaki import ja
            _misaki_ja = ja.JAG2P()
            print("[TTS] misaki[ja] G2P ready")
        except Exception as e:
            _misaki_ja = None
            print(f"[TTS] misaki[ja] unavailable ({e}); Japanese will use espeak (degraded)")
    return _misaki_ja


def _model_paths():
    onnx_env = os.environ.get("KOKORO_ONNX_PATH")
    voices_env = os.environ.get("KOKORO_VOICES_PATH")
    if onnx_env and voices_env and os.path.isfile(onnx_env) and os.path.isfile(voices_env):
        return onnx_env, voices_env
    _root = os.path.dirname(os.path.abspath(__file__))
    for base in [_root, os.path.join(_root, ".."), "/app", os.getcwd()]:
        for prefix in ["kokoro_tts", "kokoro-onnx"]:
            d = os.path.join(base, prefix) if base else prefix
            onnx = os.path.join(d, "kokoro-v1.0.onnx")
            voices = os.path.join(d, "voices-v1.0.bin")
            if os.path.isfile(onnx) and os.path.isfile(voices):
                return onnx, voices
    return "kokoro-v1.0.onnx", "voices-v1.0.bin"


def _base_lang(language):
    lang = (language or "").strip().lower()
    if not lang or lang == "auto":
        return "en"
    return lang.replace("_", "-").split("-")[0] or "en"


class _PiperViTts:
    """Offline Vietnamese TTS via sherpa-onnx VITS (Piper vi_VN model)."""

    def __init__(self):
        import sherpa_onnx

        d = os.environ.get("PIPER_VI_DIR", PIPER_VI_DEFAULT_DIR)
        model = os.environ.get("PIPER_VI_MODEL") or ""
        if not model:
            cands = sorted(glob.glob(os.path.join(d, "*.onnx")))
            model = cands[0] if cands else os.path.join(d, "vi_VN-vais1000-medium.onnx")
        tokens = os.path.join(d, "tokens.txt")
        data_dir = os.path.join(d, "espeak-ng-data")
        for p in (model, tokens, data_dir):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"[TTS][piper-vi] missing: {p}\n"
                    "Fetch it with: scripts/fetch_assets.py tts-vi"
                )
        print(f"[TTS] Loading Piper vi_VN: {model}")
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model, tokens=tokens, data_dir=data_dir
                ),
                num_threads=2,
            ),
        )
        self.tts = sherpa_onnx.OfflineTts(cfg)

    def create(self, text):
        audio = self.tts.generate(text, sid=0, speed=1.0)
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)


class TTSProcessor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        onnx_path, voices_path = _model_paths()
        print(f"[TTS] Kokoro onnx path: {onnx_path}")
        print(f"[TTS] Kokoro voices path: {voices_path}")
        if not os.path.isfile(onnx_path) or not os.path.isfile(voices_path):
            raise FileNotFoundError(f"Kokoro model not found: onnx={onnx_path}, voices={voices_path}")
        self.kokoro = Kokoro(onnx_path, voices_path)
        try:
            self.voices = set(self.kokoro.get_voices())
        except Exception:
            self.voices = set()
        self._piper_vi = None
        print(f"[TTS] Kokoro loaded OK: onnx={onnx_path}, voices={voices_path}")

    def _get_piper_vi(self):
        if self._piper_vi is None:
            try:
                self._piper_vi = _PiperViTts()
            except Exception as e:
                print(f"[TTS] Piper vi unavailable ({e}); falling back to English voice")
                self._piper_vi = False
        return self._piper_vi or None

    def _map_lang(self, language):
        full = (language or "").strip().lower()
        if not full or full == "auto":
            return "en-us"
        return LANGUAGE_MAP.get(full) or LANGUAGE_MAP.get(_base_lang(full), "en-us")

    def _voice_for(self, language):
        base = _base_lang(language)
        voice = os.environ.get(f"TTS_VOICE_{base.upper()}") or VOICE_MAP.get(base, DEFAULT_VOICE)
        if self.voices and voice not in self.voices:
            print(f"[TTS] voice {voice!r} not in model; using {DEFAULT_VOICE}")
            voice = DEFAULT_VOICE if DEFAULT_VOICE in self.voices else sorted(self.voices)[0]
        return voice

    def _phonemize(self, text, language):
        """Return (text_or_phonemes, is_phonemes)."""
        if _base_lang(language) == "ja":
            g2p = _get_misaki_ja()
            if g2p is not None:
                try:
                    phonemes, _ = g2p(text)
                    if phonemes:
                        return phonemes, True
                except Exception as e:
                    print(f"[TTS] misaki[ja] failed ({e}); using espeak lang=ja")
        return text, False

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

    def _synthesize(self, text, language):
        """Return (audio_float32, sample_rate) for the given text/language."""
        base = _base_lang(language)
        if base == "vi":
            piper = self._get_piper_vi()
            if piper is not None:
                print(f"[TTS] Speaking (piper-vi) repr={text!r}")
                return piper.create(text)

        lang = self._map_lang(language)
        voice = self._voice_for(language)
        payload, is_phonemes = self._phonemize(text, language)
        print(f"[TTS] Speaking: lang={lang} voice={voice} phonemes={is_phonemes} repr={payload!r}")
        return self.kokoro.create(
            text=payload,
            voice=voice,
            speed=1.0,
            lang=lang,
            is_phonemes=is_phonemes,
            trim=True,
        )

    def speech(self, text, language=None, output_queue=None, chunk_size=None, device_sample_rate=None, gen=None):
        # gen: barge-in generation tag. When set, each audio chunk is queued
        # as (gen, chunk) so playback can drop audio from aborted turns.
        if not text or not text.strip():
            return
        base = _base_lang(language)
        if base not in SUPPORTED_LANGS:
            print(f"[TTS] language {language!r} not supported; falling back to English")
        t0 = time.time()
        try:
            audio, sr = self._synthesize(text.strip(), language)
            self._emit(audio, sr, output_queue, chunk_size, device_sample_rate, gen)
        except Exception as e:
            print(f"[TTS] Error: {e}")
            import traceback
            traceback.print_exc()
        print(f"[TTS] Done ({time.time() - t0:.2f}s)")
