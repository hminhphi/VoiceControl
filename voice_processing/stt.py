import os
import re
import io
import json
import wave
import base64
import numpy as np
from urllib.parse import urlencode
from pydub import AudioSegment
from resampler import Resampler

STT_SAMPLE_RATE = 16000
NEMOTRON_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
OPENAI_STT_MODEL = "gpt-4o-transcribe"
OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
ELEVENLABS_STT_MODEL = "scribe_v2_realtime"
ELEVENLABS_STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
DEFAULT_OPENAI_STT_PROMPT = "The audio is entirely in English. Transcribe only in English."
WHISPER_ASR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "agent_assets", "models", "asr_whisper"
)
_WHISPER_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "whisper_trt"
)
_ELEVENLABS_ERROR_EVENTS = {
    "scribe_error",
    "scribe_auth_error",
    "scribe_quota_exceeded_error",
    "scribe_throttled_error",
    "scribe_unaccepted_terms_error",
    "scribe_rate_limited_error",
    "scribe_queue_overflow_error",
    "scribe_resource_exhausted_error",
}


def _pad_audio(audio_data, sample_width=2, channels=1):
    remainder = len(audio_data) % (sample_width * channels)
    if remainder != 0:
        padding = b"\x00" * (sample_width * channels - remainder)
        return audio_data + padding
    return audio_data


def resample_audio(audio_int16, original_sr, target_sr):
    if not isinstance(audio_int16, np.ndarray):
        audio_int16 = np.frombuffer(audio_int16, dtype=np.int16)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    if original_sr == target_sr:
        return audio_f32
    try:
        r = Resampler(original_sr, target_sr)
        samples = r.resample(audio_f32)
    except Exception:
        n = int(len(audio_int16) * target_sr / original_sr)
        samples = np.interp(
            np.linspace(0, len(audio_int16) - 1, n),
            np.arange(len(audio_int16)),
            audio_f32,
        )
    return samples


def _apply_local_whisper_patch(model_name):
    base_path = os.environ.get("WHISPER_BASE_MODEL_PATH")
    if not base_path or not os.path.isfile(base_path):
        return
    import whisper
    _orig_load = whisper.load_model

    def _local_load(name, *args, **kwargs):
        path = base_path if name == model_name else name
        return _orig_load(path, *args, **kwargs)

    whisper.load_model = _local_load


def _audio_metrics(samples):
    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return rms, peak


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sherpa_provider() -> str:
    """Execution provider for sherpa-onnx: cpu | cuda.

    Env: SHERPA_PROVIDER (falls back to ONNX_PROVIDER, default cpu).
    GPU requires a CUDA-enabled sherpa-onnx build; otherwise it falls back
    to cpu automatically (see _build_recognizer).
    """
    val = (os.environ.get("SHERPA_PROVIDER") or os.environ.get("ONNX_PROVIDER") or "cpu").strip().lower()
    if val in ("gpu", "cuda"):
        return "cuda"
    return "cpu"


def _build_recognizer(factory, **kwargs):
    """Create a sherpa recognizer with the configured provider, cpu fallback."""
    provider = _sherpa_provider()
    kwargs["provider"] = provider
    try:
        return factory(**kwargs)
    except Exception as e:
        if provider != "cpu":
            print(f"[STT] sherpa provider={provider} unavailable ({e}); using cpu", flush=True)
            kwargs["provider"] = "cpu"
            return factory(**kwargs)
        raise


# ── Whisper hallucination filtering ─────────────────────────────────────────
# Whisper invents short filler phrases ("Yeah.", "Thank you.") on noise/silence.
# These are dropped by default; disable with STT_DROP_SHORT_HALLUCINATIONS=0 and
# extend/override the list with STT_HALLUCINATION_BLOCKLIST="a,b,c".
_HALLUCINATION_TEXTS = {
    "yeah", "yeah yeah", "yep", "yup",
    "hmm", "mm", "mmm", "uh", "uhh", "um", "umm", "ah", "eh", "oh",
    "thank you", "thanks", "thank you very much",
    "thanks for watching", "thank you for watching",
    "please subscribe", "subscribe", "like and subscribe",
    "you", "bye", "bye bye", "goodbye", "hello", "hi", "ha", "haha", "wow", "huh",
    "phone beeps", "beep", "beeps", "applause", "music", "silence",
    "subtitle", "subtitles", "subs",
}
_DROP_SHORT_HALLUCINATIONS = _env_bool("STT_DROP_SHORT_HALLUCINATIONS", True)


def _strip_hallucination(text: str) -> str:
    """Return "" when the transcript looks like Whisper noise/hallucination."""
    t = (text or "").strip()
    if not t:
        return ""
    core = t.strip(" \t\r\n.!?,…。！？\"'").lower()
    if not core:
        return ""
    if core[0] in "([{" and core[-1] in ")]}":  # bracketed-only, e.g. "(phone beeps)"
        return ""
    if _DROP_SHORT_HALLUCINATIONS:
        extra = {
            w.strip().lower()
            for w in os.environ.get("STT_HALLUCINATION_BLOCKLIST", "").split(",")
            if w.strip()
        }
        if core in _HALLUCINATION_TEXTS or core in extra:
            return ""
    return t




def _normalize_optional_language(language):
    lang = (language or "").strip().lower()
    if not lang or lang == "auto":
        return None
    lang = re.split(r"[_\-.]", lang, maxsplit=1)[0]
    return lang if len(lang) in (2, 3) and lang.isalpha() else None


def _float_samples_to_int16(samples):
    samples = np.asarray(samples, dtype=np.float32)
    return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)


class WhisperTrtBackend:
    def __init__(self, device_sample_rate, model_name="small", language=None):
        self.device_sample_rate = device_sample_rate
        self.model_name = model_name
        # Empty STT_LANGUAGE/STT_LANG means auto-detect language, matching the
        # previous behavior. Set STT_LANGUAGE=en to force English decoding.
        self.language = language or None
        if os.environ.get("WHISPER_OFFLINE") or os.environ.get("WHISPER_BASE_MODEL_PATH"):
            os.environ["HF_HUB_OFFLINE"] = "1"
        from whisper_trt import load_trt_model, set_cache_dir

        trt_path = os.environ.get("WHISPER_TRT_MODEL_PATH")
        if trt_path:
            if not os.path.isfile(trt_path):
                raise FileNotFoundError(f"WHISPER_TRT_MODEL_PATH not found: {trt_path}")
            print(f"[STT] Using local TRT model: {trt_path}")
        elif os.path.isdir(_WHISPER_CACHE_DIR):
            set_cache_dir(_WHISPER_CACHE_DIR)
            print(f"[STT] Whisper TRT cache: {_WHISPER_CACHE_DIR}")
        _apply_local_whisper_patch(model_name)
        print(f"[STT] Loading Whisper TRT: {model_name} (lang={self.language or 'auto'})")
        self.model = load_trt_model(
            model_name,
            path=trt_path if trt_path else None,
            build=False,
        )
        self.model.transcribe(np.zeros(1536, dtype=np.float32), language=self.language)
        print("[STT] Whisper TRT ready")

    def transcribe(self, audio_int16):
        samples = resample_audio(audio_int16, self.device_sample_rate, STT_SAMPLE_RATE)
        result = self.model.transcribe(samples, language=self.language)
        text = result.get("text", "")
        m = re.search(r"<\|([a-z]{2})\|>", text)
        generated_lang = m.group(1) if m else None
        detected_lang = self.language or generated_lang
        clean = re.sub(r"<\|[^|]+\|>", "", text).strip()
        rms, peak = _audio_metrics(samples)
        print(
            f"[STT][debug] forced_lang={self.language or 'auto'} "
            f"generated_lang={generated_lang or 'none'} "
            f"samples={len(samples)} rms={rms:.5f} peak={peak:.5f}"
        )
        print(f"[STT][debug] raw={text!r} clean={clean!r}")
        return {
            "text": clean,
            "language": detected_lang,
            "raw_text": text,
            "forced_language": self.language,
            "generated_language": generated_lang,
            "sample_count": len(samples),
            "rms": rms,
            "peak": peak,
        }


class NemotronBackend:
    def __init__(self, device_sample_rate, model_id=None):
        self.device_sample_rate = device_sample_rate
        self.model_id = model_id or NEMOTRON_MODEL_ID
        try:
            import torch
            from transformers import AutoModelForRNNT, AutoProcessor
        except ImportError as e:
            missing = getattr(e, "name", None) or "transformers/torch"
            raise RuntimeError(
                "[STT] STT_BACKEND=nemotron requires Transformers RNNT support. "
                "Install transformers>=5.13.0 and accelerate in the voice_processing image. "
                f"Missing import: {missing}"
            ) from e

        self.torch = torch
        try:
            torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
            torch.set_num_interop_threads(int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
        except Exception:
            pass
        print(f"[STT] Loading Nemotron ASR: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForRNNT.from_pretrained(
            self.model_id,
            device_map="auto",
        )
        self.model.eval()
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sample_rate = getattr(feature_extractor, "sampling_rate", STT_SAMPLE_RATE)
        print(f"[STT] Nemotron ready sample_rate={self.sample_rate}")

    def _model_device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def _model_dtype(self):
        dtype = getattr(self.model, "dtype", None)
        if dtype is not None:
            return dtype
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return self.torch.float32

    def transcribe(self, audio_int16):
        samples = resample_audio(audio_int16, self.device_sample_rate, self.sample_rate)
        inputs = self.processor(
            samples,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model_device(), dtype=self._model_dtype())
        with self.torch.no_grad():
            output = self.model.generate(**inputs, return_dict_in_generate=True)
        text = self.processor.decode(output.sequences, skip_special_tokens=True)
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        clean = str(text).strip()
        rms, peak = _audio_metrics(samples)
        print(
            f"[STT][debug] backend=nemotron samples={len(samples)} "
            f"rms={rms:.5f} peak={peak:.5f}"
        )
        print(f"[STT][debug] raw={text!r} clean={clean!r}")
        return {
            "text": clean,
            "language": "en" if clean else None,
            "raw_text": text,
            "forced_language": None,
            "generated_language": "en" if clean else None,
            "sample_count": len(samples),
            "rms": rms,
            "peak": peak,
        }


def _normalize_openai_language(language):
    lang = (language or "").strip().lower()
    if not lang or lang == "auto":
        # OpenAI's transcription language parameter is a recognition hint, not
        # a hard language lock. Default to English because this app's voice
        # commands are expected to be English.
        return "en"
    lang = re.split(r"[_\-.]", lang, maxsplit=1)[0]
    return lang if len(lang) == 2 and lang.isalpha() else "en"


def _looks_like_local_whisper_model(model_name):
    name = (model_name or "").strip().lower()
    if not name:
        return True
    base = name.removesuffix(".en")
    return base in {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}


class OpenAITranscriptionBackend:
    def __init__(self, device_sample_rate, model_name=None, language=None):
        self.device_sample_rate = device_sample_rate
        configured_model = (
            os.environ.get("OPENAI_STT_MODEL")
            or os.environ.get("OPENAI_TRANSCRIBE_MODEL")
            or model_name
            or ""
        ).strip()
        if _looks_like_local_whisper_model(configured_model):
            configured_model = OPENAI_STT_MODEL
        self.model_name = configured_model
        self.language = _normalize_openai_language(language)
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("[STT] STT_BACKEND=openai requires OPENAI_API_KEY")
        self.url = os.environ.get("OPENAI_TRANSCRIPTIONS_URL", OPENAI_TRANSCRIPTIONS_URL).strip()
        self.timeout = float(os.environ.get("OPENAI_STT_TIMEOUT", "20"))
        configured_prompt = (
            os.environ.get("OPENAI_STT_PROMPT")
            or os.environ.get("STT_PROMPT")
            or ""
        ).strip()
        self.prompt = configured_prompt
        if not self.prompt and self.language == "en":
            self.prompt = DEFAULT_OPENAI_STT_PROMPT
        print(f"[STT] OpenAI transcription ready: {self.model_name} (lang={self.language})")

    def _to_wav_bytes(self, audio_int16):
        if not isinstance(audio_int16, np.ndarray):
            audio_int16 = np.frombuffer(audio_int16, dtype=np.int16)
        audio_int16 = np.asarray(audio_int16, dtype=np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(self.device_sample_rate))
            wav.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf, audio_int16

    def transcribe(self, audio_int16):
        wav_file, audio_int16 = self._to_wav_bytes(audio_int16)
        samples = audio_int16.astype(np.float32) / 32768.0
        rms, peak = _audio_metrics(samples)
        data = {
            "model": self.model_name,
            "response_format": "json",
            "language": self.language,
        }
        if self.prompt:
            data["prompt"] = self.prompt

        import requests

        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data,
            files={"file": ("segment.wav", wav_file, "audio/wav")},
            timeout=self.timeout,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text}
        if response.status_code >= 400:
            raise RuntimeError(f"[STT] OpenAI transcription failed: {response.status_code} {payload}")
        text = str(payload.get("text", "")).strip()
        print(
            f"[STT][debug] backend=openai model={self.model_name} "
            f"samples={len(samples)} rms={rms:.5f} peak={peak:.5f}"
        )
        print(f"[STT][debug] text={text!r}")
        return {
            "text": text,
            "language": self.language,
            "raw_text": text,
            "forced_language": self.language,
            "generated_language": self.language,
            "sample_count": len(samples),
            "rms": rms,
            "peak": peak,
        }


class ElevenLabsRealtimeTranscriptionBackend:
    def __init__(self, device_sample_rate, model_name=None, language=None, websocket_factory=None):
        self.device_sample_rate = device_sample_rate
        self.model_name = (
            os.environ.get("ELEVENLABS_STT_MODEL")
            or os.environ.get("ELEVENLABS_TRANSCRIBE_MODEL")
            or model_name
            or ELEVENLABS_STT_MODEL
        ).strip()
        if _looks_like_local_whisper_model(self.model_name):
            self.model_name = ELEVENLABS_STT_MODEL
        self.language = _normalize_optional_language(language)
        self.api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("[STT] STT_BACKEND=elevenlabs requires ELEVENLABS_API_KEY")
        self.url = os.environ.get("ELEVENLABS_STT_URL", ELEVENLABS_STT_URL).strip()
        self.timeout = float(os.environ.get("ELEVENLABS_STT_TIMEOUT", "20"))
        self.include_timestamps = _env_bool("ELEVENLABS_STT_INCLUDE_TIMESTAMPS", False)
        self.include_language_detection = _env_bool(
            "ELEVENLABS_STT_INCLUDE_LANGUAGE_DETECTION",
            bool(self.include_timestamps),
        )
        self.no_verbatim = _env_bool("ELEVENLABS_STT_NO_VERBATIM", False)
        self.commit_strategy = (
            os.environ.get("ELEVENLABS_STT_COMMIT_STRATEGY", "manual").strip().lower()
            or "manual"
        )
        if self.commit_strategy not in {"manual", "vad"}:
            self.commit_strategy = "manual"
        self.websocket_factory = websocket_factory
        print(
            f"[STT] ElevenLabs realtime ready: {self.model_name} "
            f"(lang={self.language or 'auto'})"
        )

    def _url_with_query(self):
        params = {
            "model_id": self.model_name,
            "audio_format": "pcm_16000",
            "include_timestamps": str(self.include_timestamps).lower(),
            "include_language_detection": str(self.include_language_detection).lower(),
            "commit_strategy": self.commit_strategy,
            "no_verbatim": str(self.no_verbatim).lower(),
        }
        if self.language:
            params["language_code"] = self.language
        sep = "&" if "?" in self.url else "?"
        return f"{self.url}{sep}{urlencode(params)}"

    def _connect(self):
        if self.websocket_factory is not None:
            return self.websocket_factory(
                self._url_with_query(),
                header=[f"xi-api-key: {self.api_key}"],
                timeout=self.timeout,
            )
        import websocket

        return websocket.create_connection(
            self._url_with_query(),
            header=[f"xi-api-key: {self.api_key}"],
            timeout=self.timeout,
        )

    def _to_pcm16_16k(self, audio_int16):
        samples = resample_audio(audio_int16, self.device_sample_rate, STT_SAMPLE_RATE)
        return _float_samples_to_int16(samples), samples

    def _error_message(self, payload):
        message = payload.get("message") or payload.get("error") or payload.get("detail") or payload
        return f"[STT] ElevenLabs transcription failed: {payload.get('message_type')} {message}"

    def transcribe(self, audio_int16):
        pcm16, samples = self._to_pcm16_16k(audio_int16)
        rms, peak = _audio_metrics(samples)
        message = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm16.tobytes()).decode("ascii"),
            "sample_rate": STT_SAMPLE_RATE,
            "commit": True,
        }

        ws = self._connect()
        text = ""
        detected_lang = self.language
        latest_partial = ""
        try:
            ws.send(json.dumps(message))
            while True:
                raw = ws.recv()
                if raw is None:
                    break
                payload = json.loads(raw)
                message_type = payload.get("message_type")
                if message_type in _ELEVENLABS_ERROR_EVENTS:
                    raise RuntimeError(self._error_message(payload))
                if message_type == "partial_transcript":
                    latest_partial = str(payload.get("text", "")).strip()
                    continue
                if message_type in (
                    "committed_transcript",
                    "committed_transcript_with_timestamps",
                ):
                    text = str(payload.get("text", "")).strip()
                    detected_lang = payload.get("language_code") or detected_lang
                    break
                if message_type == "session_started":
                    continue
        finally:
            try:
                ws.close()
            except Exception:
                pass

        if not text and latest_partial:
            text = latest_partial
        print(
            f"[STT][debug] backend=elevenlabs model={self.model_name} "
            f"samples={len(samples)} rms={rms:.5f} peak={peak:.5f}"
        )
        print(f"[STT][debug] text={text!r}")
        return {
            "text": text,
            "language": detected_lang,
            "raw_text": text,
            "forced_language": self.language,
            "generated_language": detected_lang,
            "sample_count": len(samples),
            "rms": rms,
            "peak": peak,
        }


class SherpaOnnxBackend:
    """
    Offline ASR via sherpa-onnx — SenseVoice or Paraformer model.

    SenseVoice supports zh/en/ja/ko/yue with INT8 quantization.
    Paraformer is faster but Chinese-only.

    Environment:
        STT_MODEL_DIR   path to model dir (model.int8.onnx + tokens.txt)
        STT_MODEL_TYPE  sense_voice | paraformer  (default: sense_voice)
    """

    def __init__(self, device_sample_rate: int, language: str | None = None):
        self.device_sample_rate = device_sample_rate
        configured_lang = os.environ.get("STT_LANGUAGE") or os.environ.get("STT_LANG") or language
        self.language = configured_lang or "auto"

        import sherpa_onnx

        model_dir = os.environ.get(
            "STT_MODEL_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_assets", "models", "asr"),
        )
        model_type = os.environ.get("STT_MODEL_TYPE", "sense_voice").strip().lower()
        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[STT][sherpa] model not found: {model_path}\n"
                "Run download_models.py or set STT_MODEL_DIR correctly."
            )
        if not os.path.isfile(tokens_path):
            raise FileNotFoundError(f"[STT][sherpa] tokens.txt not found: {tokens_path}")

        print(f"[STT] Loading sherpa-onnx {model_type}: {model_dir} (lang={self.language})", flush=True)

        lang_arg = "" if self.language in ("auto", None, "") else self.language

        if model_type == "paraformer":
            self._model = _build_recognizer(
                sherpa_onnx.OfflineRecognizer.from_paraformer,
                paraformer=model_path,
                tokens=tokens_path,
                num_threads=2,
                sample_rate=STT_SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                debug=False,
            )
        else:  # sense_voice (default)
            self._model = _build_recognizer(
                sherpa_onnx.OfflineRecognizer.from_sense_voice,
                model=model_path,
                tokens=tokens_path,
                num_threads=2,
                sample_rate=STT_SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                debug=False,
                language=lang_arg,
                use_itn=True,
            )

        print(f"[STT] sherpa-onnx ready (type={model_type})", flush=True)

    def transcribe(self, audio_int16: np.ndarray) -> dict:
        samples = resample_audio(audio_int16, self.device_sample_rate, STT_SAMPLE_RATE)
        rms, peak = _audio_metrics(samples)

        s = self._model.create_stream()
        s.accept_waveform(STT_SAMPLE_RATE, samples)
        self._model.decode_stream(s)
        text = s.result.text.strip()

        print(
            f"[STT][debug] backend=sherpa_onnx samples={len(samples)} "
            f"rms={rms:.5f} peak={peak:.5f}"
        )
        try:
            print(f"[STT][debug] text={text!r}")
        except Exception:
            print(f"[STT][debug] text={text.encode('utf-8', errors='replace')!r}")
        return {
            "text": text,
            "language": self.language,
            "raw_text": text,
            "forced_language": self.language if self.language != "auto" else None,
            "generated_language": self.language,
            "sample_count": len(samples),
            "rms": rms,
            "peak": peak,
        }


class SherpaWhisperBackend:
    """Offline multilingual ASR via sherpa-onnx Whisper.

    A single Whisper model covers vi/en/ja (and more) and reports the detected
    language, so it also drives TTS language selection. Files come from
    ``csukuangfj/sherpa-onnx-whisper-<size>``:
        <prefix>-encoder.int8.onnx, <prefix>-decoder.int8.onnx, <prefix>-tokens.txt

    Environment:
        STT_WHISPER_DIR    model directory (default: agent_assets/models/asr_whisper)
        STT_WHISPER_MODEL  filename prefix, e.g. small/tiny/base (default: small)
        STT_WHISPER_INT8   use int8 weights (default: 1)
        STT_LANGUAGE       force a language; empty/auto = auto-detect
    """

    def __init__(self, device_sample_rate: int, language: str | None = None):
        self.device_sample_rate = device_sample_rate

        import sherpa_onnx

        model_dir = os.environ.get(
            "STT_WHISPER_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "agent_assets", "models", "asr_whisper"),
        )
        prefix = (os.environ.get("STT_WHISPER_MODEL", "small") or "small").strip()
        suffix = ".int8.onnx" if _env_bool("STT_WHISPER_INT8", True) else ".onnx"
        encoder = os.path.join(model_dir, f"{prefix}-encoder{suffix}")
        decoder = os.path.join(model_dir, f"{prefix}-decoder{suffix}")
        tokens = os.path.join(model_dir, f"{prefix}-tokens.txt")
        for p in (encoder, decoder, tokens):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"[STT][whisper] model file not found: {p}\n"
                    "Fetch it with: scripts/fetch_assets.py asr-whisper"
                )

        lang = (
            os.environ.get("STT_LANGUAGE")
            or os.environ.get("STT_LANG")
            or language
            or ""
        ).strip().lower()
        self.language = "" if lang in ("", "auto") else lang

        print(
            f"[STT] Loading sherpa-onnx Whisper ({prefix}): {model_dir} "
            f"(lang={self.language or 'auto'})",
            flush=True,
        )
        self._model = _build_recognizer(
            sherpa_onnx.OfflineRecognizer.from_whisper,
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            language=self.language,
            task="transcribe",
            num_threads=2,
            debug=False,
        )
        print("[STT] sherpa-onnx Whisper ready", flush=True)

    def transcribe(self, audio_int16):
        samples = resample_audio(audio_int16, self.device_sample_rate, STT_SAMPLE_RATE)
        rms, peak = _audio_metrics(samples)

        s = self._model.create_stream()
        s.accept_waveform(STT_SAMPLE_RATE, samples)
        self._model.decode_stream(s)
        res = s.result
        text = _strip_hallucination(getattr(res, "text", "") or "")
        lang = getattr(res, "lang", None) or None

        print(
            f"[STT][debug] backend=sherpa_whisper samples={len(samples)} "
            f"rms={rms:.5f} peak={peak:.5f} lang={lang}"
        )
        try:
            print(f"[STT][debug] text={text!r}")
        except Exception:
            pass
        return {
            "text": text,
            "language": lang,
            "raw_text": text,
            "forced_language": self.language or None,
            "generated_language": lang,
            "sample_count": len(samples),
            "rms": rms,
            "peak": peak,
        }


class FasterWhisperBackend:
    """Multilingual ASR via faster-whisper (CTranslate2) — accurate + GPU.

    Recommended for vi/en/ja accuracy on a GPU (e.g. large-v3). Auto-detects
    language and returns it (drives the LLM/TTS language).

    Environment:
        FASTER_WHISPER_MODEL    large-v3 | medium | small ... (default large-v3)
        FASTER_WHISPER_DEVICE   auto | cuda | cpu (default auto)
        FASTER_WHISPER_COMPUTE  float16 | int8_float16 | int8 | float32 (auto)
        FASTER_WHISPER_BEAM     beam size (default 5)
        STT_LANGUAGE            force a language; empty/auto = auto-detect
    """

    def __init__(self, device_sample_rate: int, language: str | None = None):
        from faster_whisper import WhisperModel

        self.device_sample_rate = device_sample_rate
        model_size = (os.environ.get("FASTER_WHISPER_MODEL", "large-v3") or "large-v3").strip()
        device = (os.environ.get("FASTER_WHISPER_DEVICE", "auto") or "auto").strip().lower()
        compute = (os.environ.get("FASTER_WHISPER_COMPUTE", "") or "").strip().lower()
        if not compute:
            compute = "float16" if device in ("cuda", "auto") else "int8"

        lang = (
            os.environ.get("STT_LANGUAGE") or os.environ.get("STT_LANG") or language or ""
        ).strip().lower()
        self.language = "" if lang in ("", "auto") else lang
        self.beam_size = int(os.environ.get("FASTER_WHISPER_BEAM", "5"))

        print(
            f"[STT] Loading faster-whisper {model_size} device={device} "
            f"compute={compute} (lang={self.language or 'auto'})",
            flush=True,
        )
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute)
            # Warmup: CUDA libs (cuBLAS/cuDNN) are loaded lazily on first run, so
            # force a tiny decode now to surface failures and fall back to CPU.
            list(self.model.transcribe(np.zeros(1600, dtype=np.float32))[0])
        except Exception as e:
            if device != "cpu":
                print(f"[STT] faster-whisper {device}/{compute} failed ({e}); retry cpu/int8", flush=True)
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            else:
                raise
        print("[STT] faster-whisper ready", flush=True)

    def transcribe(self, audio_int16):
        samples = resample_audio(audio_int16, self.device_sample_rate, STT_SAMPLE_RATE)
        rms, peak = _audio_metrics(samples)

        no_speech_threshold = float(os.environ.get("FW_NO_SPEECH_THRESHOLD", "0.6"))
        log_prob_threshold = float(os.environ.get("FW_LOG_PROB_THRESHOLD", "-1.0"))
        min_avg_logprob = float(os.environ.get("FW_MIN_AVG_LOGPROB", "-1.0"))

        segments, info = self.model.transcribe(
            samples,
            language=self.language or None,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=2.4,
            temperature=0.0,
        )
        texts = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0.0) > no_speech_threshold:
                continue
            if getattr(seg, "avg_logprob", 0.0) < min_avg_logprob:
                continue
            part = _strip_hallucination(seg.text)
            if part:
                texts.append(part)
        text = " ".join(texts).strip()
        lang = getattr(info, "language", None)
        prob = getattr(info, "language_probability", None)

        print(
            f"[STT][debug] backend=faster_whisper samples={len(samples)} "
            f"rms={rms:.5f} peak={peak:.5f} lang={lang} prob={prob} "
            f"kept={len(texts)}"
        )
        try:
            print(f"[STT][debug] text={text!r}")
        except Exception:
            pass
        return {
            "text": text,
            "language": lang,
            "raw_text": text,
            "forced_language": self.language or None,
            "generated_language": lang,
            "sample_count": len(samples),
            "rms": rms,
            "peak": peak,
        }


class STTProcessor:
    def __init__(self, device_sample_rate, model_name="small", language=None):
        # Unified default: Whisper on GPU. `whisper` resolves to the best GPU
        # Whisper runtime for the platform (TensorRT on Jetson, faster-whisper
        # CUDA on x86) so PC dev and Jetson share one configuration.
        backend = (os.environ.get("STT_BACKEND") or "whisper").strip().lower()
        backend = backend.replace("-", "_")
        if backend == "whisper":
            self.backend_name, self.backend = self._build_whisper(
                device_sample_rate, model_name, language
            )
        elif backend in ("whisper_trt", "trt"):
            try:
                self.backend_name = "whisper_trt"
                self.backend = WhisperTrtBackend(device_sample_rate, model_name, language)
            except (ImportError, ModuleNotFoundError) as e:
                print(f"[STT] whisper_trt not available ({e}); falling back to sherpa_onnx", flush=True)
                self.backend_name = "sherpa_onnx"
                self.backend = SherpaOnnxBackend(device_sample_rate, language=language)
        elif backend in ("nemotron", "nemotron_en", "nemotron_speech"):
            self.backend_name = "nemotron"
            model_id = os.environ.get("NEMOTRON_MODEL_ID", NEMOTRON_MODEL_ID).strip()
            self.backend = NemotronBackend(device_sample_rate, model_id=model_id)
        elif backend in ("openai", "openai_whisper", "openai_transcribe"):
            self.backend_name = "openai"
            self.backend = OpenAITranscriptionBackend(device_sample_rate, model_name, language)
        elif backend in ("elevenlabs", "elevenlabs_realtime", "scribe"):
            self.backend_name = "elevenlabs"
            self.backend = ElevenLabsRealtimeTranscriptionBackend(
                device_sample_rate,
                model_name,
                language,
            )
        elif backend in ("faster_whisper", "fasterwhisper", "ct2", "whisper_ct2"):
            try:
                self.backend_name = "faster_whisper"
                self.backend = FasterWhisperBackend(device_sample_rate, language=language)
            except (ImportError, ModuleNotFoundError) as e:
                print(f"[STT] faster_whisper unavailable ({e}); trying sherpa_whisper", flush=True)
                try:
                    self.backend_name = "sherpa_whisper"
                    self.backend = SherpaWhisperBackend(device_sample_rate, language=language)
                except (ImportError, ModuleNotFoundError, FileNotFoundError) as e2:
                    print(f"[STT] sherpa_whisper unavailable ({e2}); using sherpa_onnx", flush=True)
                    self.backend_name = "sherpa_onnx"
                    self.backend = SherpaOnnxBackend(device_sample_rate, language=language)
        elif backend in ("sherpa_whisper", "whisper_sherpa", "whisper_onnx", "whisper_multilingual"):
            self.backend_name = "sherpa_whisper"
            self.backend = SherpaWhisperBackend(device_sample_rate, language=language)
        elif backend in ("sherpa_onnx", "sherpa", "sense_voice", "sensevoice"):
            self.backend_name = "sherpa_onnx"
            self.backend = SherpaOnnxBackend(device_sample_rate, language=language)
        else:
            raise ValueError(
                f"Unsupported STT_BACKEND={backend!r}; "
                "expected whisper (GPU), whisper_trt, faster_whisper, sherpa_whisper, nemotron, openai, elevenlabs, or sherpa_onnx"
            )

    @staticmethod
    def _build_whisper(device_sample_rate, model_name, language):
        """Resolve `STT_BACKEND=whisper` to a GPU Whisper implementation.

        Preference (GPU-first, per platform):
          1. TensorRT Whisper (Jetson `whisper_trt`)   — GPU
          2. faster-whisper (CTranslate2 CUDA)          — GPU (x86)
          3. faster-whisper CPU
          4. sherpa-onnx Whisper -> SenseVoice
        """
        try:
            import tensorrt  # noqa: F401
            try:
                return "whisper_trt", WhisperTrtBackend(device_sample_rate, model_name, language)
            except Exception as e:
                print(f"[STT] whisper_trt unavailable ({e}); trying faster-whisper", flush=True)
        except Exception:
            pass
        try:
            return "faster_whisper", FasterWhisperBackend(device_sample_rate, language=language)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[STT] faster_whisper unavailable ({e}); trying sherpa_whisper", flush=True)
        try:
            return "sherpa_whisper", SherpaWhisperBackend(device_sample_rate, language=language)
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
            print(f"[STT] sherpa_whisper unavailable ({e}); using sherpa_onnx", flush=True)
        return "sherpa_onnx", SherpaOnnxBackend(device_sample_rate, language=language)

    def transcribe(self, audio_int16):
        return self.backend.transcribe(audio_int16)
