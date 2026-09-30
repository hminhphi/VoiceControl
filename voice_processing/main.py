#!/usr/bin/env python3
"""
Standalone pipeline: Wake word -> VAD -> STT -> Orchestrator (stream) -> TTS.
Orchestrator tokens are buffered every n chunks (default 5), each segment is sent to TTS.
"""
import os
from runtime_tuning import apply_environment_thread_limits, configure_native_runtime

apply_environment_thread_limits()

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import queue
import time
import uuid
from datetime import datetime
import numpy as np
import soundfile as sf
import sounddevice as sd
from threading import Thread, Event, get_native_id

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from log_utils import install_timestamped_print

install_timestamped_print()
configure_native_runtime()

from audio_utils import (
    CircularBuffer,
    get_current_pulse_playback_sink,
    get_current_pulse_record_source,
    get_hostapi_name,
    get_pulse_default_sink,
    get_pulse_default_source,
    get_valid_input_device_id,
)
from notification_sounds import play_into_queue as play_notification
from wake_word import WakeWordProcessor
from vad import VADProcessor
from stt import STTProcessor
from tts import TTSProcessor
from lang import detect_lang
from speaker import get_analyzer, clarify_message
from tse import get_extractor
from orchestrator_client import send_and_stream
from aec import AecEngine



def _env_int(name, default, minimum=1):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
        if value < minimum:
            raise ValueError
        return value
    except ValueError:
        print(f"[audio][config] invalid {name}={raw!r}; using {default}", flush=True)
        return default


def _env_float(name, default, minimum=0.0):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
        if value < minimum:
            raise ValueError
        return value
    except ValueError:
        print(f"[audio][config] invalid {name}={raw!r}; using {default}", flush=True)
        return default


def _audio_latency_from_env(name, default="high"):
    raw = os.environ.get(name, default).strip().lower()
    if raw in ("low", "high"):
        return raw
    try:
        latency = float(raw)
        if latency < 0:
            raise ValueError
        return latency
    except ValueError:
        print(f"[audio][config] invalid {name}={raw!r}; using {default!r}", flush=True)
        return default


DEVICE_SAMPLE_RATE = _env_int("AUDIO_SAMPLE_RATE", 48000)
CHANNELS = 1
CHUNK = _env_int("AUDIO_CHUNK", 2048)
AUDIO_LATENCY = _audio_latency_from_env("AUDIO_LATENCY", "0.08")
AUDIO_STATUS_LOG_INTERVAL = _env_float("AUDIO_STATUS_LOG_INTERVAL", 2.0)
AUDIO_LEVEL_LOG_INTERVAL = _env_float("AUDIO_LEVEL_LOG_INTERVAL", 1)
AUDIO_MIC_LOG_DBFS = _env_float("AUDIO_MIC_LOG_DBFS", -50.0, minimum=-120.0)
AUDIO_SPK_LOG_DBFS = _env_float("AUDIO_SPK_LOG_DBFS", -60.0, minimum=-120.0)
VAD_FIRST_TIMEOUT = _env_float("VAD_FIRST_TIMEOUT", 8.0)
SEGMENT_SILENCE = _env_float("SEGMENT_SILENCE", 1.0)
TURN_END_SILENCE = _env_float("TURN_END_SILENCE", 2.5)
MIN_SEGMENT_SEC = _env_float("MIN_SEGMENT_SEC", 0.095)
SPEECH_PREROLL_SEC = _env_float("SPEECH_PREROLL_SEC", 0.8)
LISTENING_GRACE_SEC = _env_float("LISTENING_GRACE_SEC", 0.45)
# Shorter grace for follow-up entries: TTS echo is already cancelled by AEC,
# and users start speaking immediately (long grace discards onset syllables).
FOLLOWUP_GRACE_SEC = _env_float("FOLLOWUP_GRACE_SEC", 0.2)
WAKE_WORD_DEBOUNCE = 0.8
BUFFER_MAX = 300 * CHUNK
TTS_CHUNK_COUNT = int(os.environ.get("TTS_CHUNK_COUNT", "5"))
VAD_DETECT_THRESHOLD = _env_float("VAD_DETECT_THRESHOLD", 0.550)
# Drop near-silence / noise-only segments before they reach STT (0 = disabled).
try:
    STT_MIN_RMS = float(os.environ.get("STT_MIN_RMS", "0.0") or "0.0")
except ValueError:
    STT_MIN_RMS = 0.0

# Speaker overlap stage (opt-in, GPU): diarize each turn, detect overlapped
# speech, and optionally ask to repeat instead of feeding mixed audio to the LLM.
SPEAKER_ENABLED = os.environ.get("SPEAKER_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
SPEAKER_GATE_OVERLAP = os.environ.get("SPEAKER_GATE_OVERLAP", "0").strip().lower() in ("1", "true", "yes", "on")
SPEAKER_OVERLAP_MIN = float(os.environ.get("SPEAKER_OVERLAP_MIN", "0.15") or "0.15")
SPEAKER_TSE_ENABLED = os.environ.get("SPEAKER_TSE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
FOLLOWUP_LISTEN_SEC = _env_float("FOLLOWUP_LISTEN_SEC", 12.0)
# Barge-in (xiaozhi-style interruption): keep VAD listening during TTS
# replies; sustained speech aborts playback and starts a new turn.
BARGE_IN_ENABLED = os.environ.get("BARGE_IN_ENABLED", "1").strip() in ("1", "true", "yes")
BARGE_IN_MIN_PLAY_SEC = _env_float("BARGE_IN_MIN_PLAY_SEC", 1.0)
VAD_LOG_THRESHOLD = 0.400
VAD_LOG_INTERVAL = 0.5
AUDIO_STREAM_RETRY_SEC = float(os.environ.get("AUDIO_STREAM_RETRY_SEC", "3.0"))

STT_DEBUG_OUTPUT_DIR = os.environ.get(
    "STT_DEBUG_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)


def _audio_stats(audio_int16):
    if audio_int16 is None or len(audio_int16) == 0:
        return 0.0, 0.0
    samples = audio_int16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    return rms, peak


def _concat_int16(parts):
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return np.array([], dtype=np.int16)
    return np.concatenate(parts)


def _to_f32_16k(audio_int16, src_sr):
    f = audio_int16.astype(np.float32) / 32768.0
    if src_sr == 16000:
        return f
    try:
        from resampler import Resampler
        return Resampler(src_sr, 16000).resample(f)
    except Exception:
        n = int(len(f) * 16000 / src_sr)
        return np.interp(np.linspace(0, len(f) - 1, n), np.arange(len(f)), f).astype(np.float32)


def _dbfs(value):
    if value <= 0:
        return -120.0
    return max(20.0 * float(np.log10(value)), -120.0)


def _audio_level_stats(samples):
    if samples is None or len(samples) == 0:
        return 0.0, 0.0, -120.0, -120.0
    arr = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(arr * arr)))
    peak = float(np.max(np.abs(arr)))
    return rms, peak, _dbfs(rms), _dbfs(peak)


def _save_debug_segment(audio_int16, prefix, duration):
    try:
        os.makedirs(STT_DEBUG_OUTPUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(STT_DEBUG_OUTPUT_DIR, f"{prefix}_{stamp}_{duration:.2f}s.wav")
        sf.write(path, audio_int16, DEVICE_SAMPLE_RATE, subtype="PCM_16")
        return path
    except Exception as e:
        print(f"[STT][debug] failed to save segment: {e}")
def acquire_single_instance_lock(port=48199):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        return s
    except OSError:
        print(f"[voice_processing] WARNING: Another voice_processing instance is already running (lock port {port} in use). Exiting to prevent duplicate requests.", flush=True)
        sys.exit(0)


def run():
    _lock = acquire_single_instance_lock()
    def select_stream_device():
        preferred = os.getenv("MIC_DEVICE_NAME")
        dev_id, dev_info = get_valid_input_device_id(preferred)
        if dev_id is None:
            dev_id = sd.default.device[0]
            dev_info = sd.query_devices(dev_id)
        if dev_id is None or dev_info is None:
            raise RuntimeError("No valid audio input device found")

        device_name = dev_info.get("name", "?")
        hostapi_name = get_hostapi_name(dev_info)
        pulse_source = get_pulse_default_source()
        pulse_sink = get_pulse_default_sink()
        try:
            default_out = sd.default.device[1]
        except Exception:
            default_out = None

        if dev_info.get("max_output_channels", 0) >= 1:
            out_id = dev_id
        elif default_out is not None and default_out >= 0:
            out_id = default_out
        else:
            out_id = dev_id
        sd.default.device = (dev_id, out_id)
        print(
            f"[audio] stream input id={dev_id} name={device_name!r} "
            f"output id={out_id} hostapi={hostapi_name!r} preferred={preferred!r} "
            f"pulse_default_source={pulse_source!r} "
            f"pulse_default_sink={pulse_sink!r}",
            flush=True,
        )
        return dev_id, out_id

    ww = WakeWordProcessor(DEVICE_SAMPLE_RATE)
    vad = VADProcessor(DEVICE_SAMPLE_RATE)
    # Leave STT_LANGUAGE/STT_LANG empty to let Whisper auto-detect. STT_MODEL
    # defaults to small.en for English-only commands.
    stt_lang = os.environ.get("STT_LANGUAGE") or os.environ.get("STT_LANG") or None
    if stt_lang and len(stt_lang) > 2:
        stt_lang = stt_lang.split("_")[0].split("-")[0]
    stt_model = os.environ.get("STT_MODEL", "small.en")
    stt = STTProcessor(DEVICE_SAMPLE_RATE, model_name=stt_model, language=stt_lang or None)
    tts = TTSProcessor(DEVICE_SAMPLE_RATE)

    # AEC engine: WebRTC echo + noise cancellation
    # Runs before wake word + VAD. Fails gracefully if webrtc_apm unavailable.
    aec_delay_ms = int(os.environ.get("AEC_DELAY_MS", "60"))
    aec_noise_suppress = os.environ.get("AEC_NOISE_SUPPRESS", "1").strip() in ("1", "true", "yes")
    aec = AecEngine(
        near_rate=DEVICE_SAMPLE_RATE,
        far_rate=DEVICE_SAMPLE_RATE,
        delay_ms=aec_delay_ms,
        enable_preprocess=aec_noise_suppress,
    )


    input_buffer = CircularBuffer(maxsize=BUFFER_MAX)
    output_queue = queue.Queue()
    stt_queue = queue.Queue(maxsize=5)
    audio_log_queue = queue.Queue(maxsize=100)
    # TTS work queue: decouples LLM token reception (WS thread) from
    # blocking Kokoro synthesis so the stream never stalls mid-sentence.
    tts_text_queue = queue.Queue()
    stop = Event()
    logged_audio_callback_tid = False

    END_TURN = object()

    ww_detected = False
    vad_detected_once = False
    timeout_at = None
    last_voice_time = None
    pending_turn = False
    collected_speech = np.array([], dtype=np.int16)
    pre_speech_audio = np.array([], dtype=np.int16)
    ignore_audio_until = 0.0
    last_wake_word_time = 0.0
    last_vad_print_time = 0.0
    is_tts_playing = False
    # Follow-up conversation window (no wake-word needed after a TTS reply).
    followup_until = None
    active_session_id = None
    # Barge-in generation counter: bumped on every interruption; TTS audio
    # tagged with an older generation is dropped at playback.
    turn_gen = 0
    tts_play_start = None

    def enqueue_audio_log(event):
        try:
            audio_log_queue.put_nowait(event)
        except queue.Full:
            pass

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal logged_audio_callback_tid
        if not logged_audio_callback_tid:
            enqueue_audio_log(("callback_tid", get_native_id()))
            logged_audio_callback_tid = True
        if status:
            enqueue_audio_log(("status", time.time(), str(status)))

        # ── Output side: get playback chunk ──────────────────
        # TTS audio is tagged (gen, chunk); chunks from an aborted
        # (barged-in) generation are dropped so stale replies never play.
        try:
            playback_chunk = None
            while playback_chunk is None:
                item = output_queue.get_nowait()
                if isinstance(item, tuple):
                    g, c = item
                    if g == turn_gen:
                        playback_chunk = c
                    # else: stale generation after barge-in → drop, try next
                else:
                    playback_chunk = item  # raw beep chunk: always play
            outdata[:] = playback_chunk
        except queue.Empty:
            outdata.fill(0)

        # Feed output to AEC far-end reference (fast path — no resampling here)
        aec.feed_far(outdata)

        # ── Input side: AEC near-end processing ──────────────
        near_f32 = indata[:, 0].astype(np.float32)
        if aec.active:
            near_f32 = aec.process_near(near_f32)

        chunk = (near_f32 * 32768).astype(np.int16)
        input_buffer.put(chunk)


    def process_audio_logs():
        last_input_status_log = 0.0

        print(f"[audio][thread] process_audio_logs_tid={get_native_id()}", flush=True)
        while not stop.is_set():
            try:
                event = audio_log_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            kind = event[0]
            if kind == "callback_tid":
                print(f"[audio][thread] callback_tid={event[1]}", flush=True)
            elif kind == "status":
                event_time, status_text = event[1], event[2]
                if event_time - last_input_status_log >= AUDIO_STATUS_LOG_INTERVAL:
                    print(f"[audio][status] {status_text}", flush=True)
                    last_input_status_log = event_time

    def process_audio():
        nonlocal ww_detected, vad_detected_once, timeout_at, last_voice_time, pending_turn, collected_speech, pre_speech_audio, ignore_audio_until, last_wake_word_time, last_vad_print_time, is_tts_playing, followup_until, active_session_id, turn_gen, tts_play_start

        print(f"[audio][thread] process_audio_tid={get_native_id()}", flush=True)
        while not stop.is_set():
            try:
                chunk = input_buffer.get(CHUNK)
                now = time.time()
                if chunk is not None and not is_tts_playing:
                    ww.update_state(chunk.tobytes())
                    if ww.get_detected() and (now - last_wake_word_time) >= WAKE_WORD_DEBOUNCE:
                        last_wake_word_time = now
                        ww_detected = True
                        timeout_at = now + VAD_FIRST_TIMEOUT
                        vad_detected_once = False
                        last_voice_time = None
                        pending_turn = False
                        collected_speech = np.array([], dtype=np.int16)
                        pre_speech_audio = np.array([], dtype=np.int16)
                        ignore_audio_until = now + LISTENING_GRACE_SEC
                        vad.empty_buffer()
                        # Fresh wake-word starts a new conversation.
                        followup_until = None
                        active_session_id = None
                        print("[wake_word] Accepted; listening for command")
                        play_notification(
                            "listening",
                            output_queue,
                            CHUNK,
                            DEVICE_SAMPLE_RATE,
                        )
                    # Follow-up window: no wake-word needed right after a reply.
                    if not ww_detected and followup_until is not None and not is_tts_playing:
                        if now >= followup_until:
                            print("[followup] Window expired; back to wake-word mode")
                            followup_until = None
                            active_session_id = None
                        else:
                            ww_detected = True
                            timeout_at = followup_until
                            vad_detected_once = False
                            last_voice_time = None
                            pending_turn = False
                            collected_speech = np.array([], dtype=np.int16)
                            pre_speech_audio = np.array([], dtype=np.int16)
                            ignore_audio_until = now + FOLLOWUP_GRACE_SEC
                            vad.empty_buffer()
                            print(
                                f"[followup] Listening for follow-up "
                                f"({followup_until - now:.0f}s left, no wake-word needed)"
                            )
                            play_notification(
                                "listening",
                                output_queue,
                                CHUNK,
                                DEVICE_SAMPLE_RATE,
                            )
                    if ww_detected and now >= ignore_audio_until:
                        vad.update_buffer(chunk.tobytes())
                        if vad_detected_once:
                            collected_speech = np.concatenate((collected_speech, chunk))
                        else:
                            pre_speech_audio = np.concatenate((pre_speech_audio, chunk))
                            max_preroll = int(SPEECH_PREROLL_SEC * DEVICE_SAMPLE_RATE)
                            if len(pre_speech_audio) > max_preroll:
                                pre_speech_audio = pre_speech_audio[-max_preroll:]
                elif chunk is not None and is_tts_playing and BARGE_IN_ENABLED:
                    # ── Barge-in: VAD stays live during TTS playback ──
                    # AEC far-end reference already cancels speaker echo, so
                    # sustained voice here means the user is interrupting.
                    vad.update_buffer(chunk.tobytes())
                    pre_speech_audio = np.concatenate((pre_speech_audio, chunk))
                    max_preroll = int(SPEECH_PREROLL_SEC * DEVICE_SAMPLE_RATE)
                    if len(pre_speech_audio) > max_preroll:
                        pre_speech_audio = pre_speech_audio[-max_preroll:]
                    p = vad.get_prob()
                    if p > VAD_LOG_THRESHOLD and (now - last_vad_print_time) >= VAD_LOG_INTERVAL:
                        print(f"[VAD][barge] prob={p:.3f} threshold={VAD_DETECT_THRESHOLD:.3f}")
                        last_vad_print_time = now
                    play_dur = now - (tts_play_start or now)
                    if vad.is_speech() and play_dur >= BARGE_IN_MIN_PLAY_SEC:
                        # Abort playback now; stale TTS audio tagged with the
                        # old generation is dropped at playback/output.
                        turn_gen += 1
                        try:
                            while True:
                                output_queue.get_nowait()
                        except queue.Empty:
                            pass
                        print("[barge-in] Interrupted; aborted playback, listening")
                        play_notification(
                            "listening",
                            output_queue,
                            CHUNK,
                            DEVICE_SAMPLE_RATE,
                        )
                        ww_detected = True
                        timeout_at = now + VAD_FIRST_TIMEOUT
                        # VAD is already hot and speech is ongoing: the new
                        # turn starts mid-speech with onset kept from pre-roll.
                        vad_detected_once = True
                        last_voice_time = now
                        pending_turn = False
                        collected_speech = pre_speech_audio.copy()
                        pre_speech_audio = np.array([], dtype=np.int16)
                        ignore_audio_until = now
                        is_tts_playing = False
                        tts_play_start = None
                        print("[VAD] Speech start (barge-in); capturing command")

                if ww_detected:
                    p = vad.get_prob()
                    if p > VAD_LOG_THRESHOLD and (now - last_vad_print_time) >= VAD_LOG_INTERVAL:
                        print(f"[VAD] prob={p:.3f} threshold={VAD_DETECT_THRESHOLD:.3f}")
                        last_vad_print_time = now
                    if p > VAD_DETECT_THRESHOLD:
                        if not vad_detected_once:
                            vad_detected_once = True
                            collected_speech = pre_speech_audio.copy()
                            print("[VAD] Speech start; capturing command")
                        last_voice_time = now

                if last_voice_time is not None:
                    segment_at = last_voice_time + SEGMENT_SILENCE
                    turn_end_at = last_voice_time + TURN_END_SILENCE
                    if pending_turn and now >= turn_end_at:
                        try:
                            stt_queue.put(END_TURN, block=False)
                        except queue.Full:
                            pass
                        print("[TURN] End detected; sending transcript")
                        play_notification(
                            "sending",
                            output_queue,
                            CHUNK,
                            DEVICE_SAMPLE_RATE,
                        )
                        ww_detected = False
                        timeout_at = None
                        last_voice_time = None
                        pending_turn = False
                        collected_speech = np.array([], dtype=np.int16)
                        pre_speech_audio = np.array([], dtype=np.int16)
                        # Consume follow-up trigger; process_stt re-arms a fresh
                        # window after the reply finishes playing.
                        followup_until = None
                    elif now >= segment_at and len(collected_speech) > 0:
                        dur = len(collected_speech) / DEVICE_SAMPLE_RATE
                        if dur >= MIN_SEGMENT_SEC:
                            audio_segment = collected_speech.copy()
                            rms, peak = _audio_stats(audio_segment)
                            if STT_MIN_RMS > 0 and rms < STT_MIN_RMS:
                                print(
                                    f"[STT] Skip low-energy segment "
                                    f"rms={rms:.5f} < STT_MIN_RMS={STT_MIN_RMS:.5f} "
                                    f"dur={dur:.2f}s"
                                )
                            else:
                                try:
                                    stt_queue.put(audio_segment, block=False)
                                    print(
                                        f"[STT] Queued audio segment dur={dur:.2f}s "
                                        f"rms={rms:.5f} peak={peak:.5f}"
                                    )
                                except queue.Full:
                                    print("[STT] Queue full; dropped audio segment")
                            pending_turn = True
                        collected_speech = np.array([], dtype=np.int16)
                    elif now >= turn_end_at and not pending_turn:
                        print("[TURN] Timeout before speech segment")
                        play_notification(
                            "timeout",
                            output_queue,
                            CHUNK,
                            DEVICE_SAMPLE_RATE,
                        )
                        ww_detected = False
                        timeout_at = None
                        last_voice_time = None
                        collected_speech = np.array([], dtype=np.int16)
                        pre_speech_audio = np.array([], dtype=np.int16)
                        # No speech in follow-up window -> end conversation.
                        followup_until = None
                        active_session_id = None
                elif timeout_at is not None and now >= timeout_at:
                    print("[TURN] Timeout waiting for speech after wake word")
                    play_notification(
                        "timeout",
                        output_queue,
                        CHUNK,
                        DEVICE_SAMPLE_RATE,
                    )
                    ww_detected = False
                    timeout_at = None
                    collected_speech = np.array([], dtype=np.int16)
                    pre_speech_audio = np.array([], dtype=np.int16)
                    # Covers follow-up window expiry as well.
                    followup_until = None
                    active_session_id = None

                if chunk is None:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    def process_stt():
        nonlocal ignore_audio_until, is_tts_playing, followup_until, active_session_id, tts_play_start
        print(f"[audio][thread] process_stt_tid={get_native_id()}", flush=True)
        turn_texts = []
        turn_lang = None
        turn_audio_parts = []
        while not stop.is_set():
            try:
                item = stt_queue.get(timeout=1.0)
                if item is END_TURN:
                    text = " ".join(turn_texts).strip()
                    turn_texts = []
                    if not text:
                        print("[TURN] No transcript text; skip response")
                        turn_lang = None
                        continue
                    lang = turn_lang
                    print(f"[TURN] Transcript ready: {text!r} lang={lang}")
                    # ── Speaker overlap / TSE / gate (opt-in, GPU) ──
                    if (SPEAKER_ENABLED or SPEAKER_TSE_ENABLED) and turn_audio_parts:
                        try:
                            wav16 = _to_f32_16k(_concat_int16(turn_audio_parts), DEVICE_SAMPLE_RATE)
                            res = None
                            analyzer = get_analyzer() if SPEAKER_ENABLED else None
                            if analyzer is not None:
                                res = analyzer.analyze(wav16, 16000)
                                print(
                                    f"[SPEAKER] turns={len(res['turns'])} "
                                    f"overlap={res['overlap_ratio']:.2f} dominant={res['dominant']}"
                                )
                            overlap = res["overlap_ratio"] if res else 0.0
                            # Target Speaker Extraction on overlap (separation + select).
                            if SPEAKER_TSE_ENABLED and (res is None or overlap >= SPEAKER_OVERLAP_MIN):
                                ext = get_extractor()
                                if ext is not None:
                                    target = ext.extract_target(wav16, get_analyzer())
                                    ti16 = np.clip(target * 32768, -32768, 32767).astype(np.int16)
                                    out2 = stt.transcribe(ti16)
                                    nt = out2.get("text", "").strip()
                                    if nt:
                                        print(f"[TSE] target transcript: {nt!r} lang={out2.get('language')}")
                                        text = nt
                                        lang = out2.get("language") or lang
                            # Optional gate: ask to repeat when overlap and no TSE.
                            if (SPEAKER_GATE_OVERLAP and not SPEAKER_TSE_ENABLED
                                    and res and overlap >= SPEAKER_OVERLAP_MIN):
                                msg = clarify_message(lang or "en")
                                print(f"[SPEAKER] Overlap gate -> {msg}")
                                tts.speech(
                                    msg,
                                    language=lang,
                                    output_queue=output_queue,
                                    chunk_size=CHUNK,
                                    device_sample_rate=DEVICE_SAMPLE_RATE,
                                )
                                ignore_audio_until = time.time() + LISTENING_GRACE_SEC
                                followup_until = time.time() + FOLLOWUP_LISTEN_SEC
                                turn_audio_parts = []
                                turn_lang = None
                                continue
                        except Exception as e:
                            print(f"[SPEAKER][error] {type(e).__name__}: {e}", flush=True)
                    turn_audio_parts = []
                else:
                    audio = item
                    turn_audio_parts.append(audio)
                    dur = len(audio) / DEVICE_SAMPLE_RATE
                    t0 = time.time()
                    out = stt.transcribe(audio)
                    part = out.get("text", "").strip()
                    if not part:
                        saved_path = _save_debug_segment(audio, "empty_stt", dur)
                        if saved_path:
                            print(f"[STT][debug] saved empty segment: {saved_path}")
                    if part:
                        turn_texts.append(part)
                    seg_lang = out.get("language")
                    if not part:
                        # Empty (hallucination-dropped) segments must not set the
                        # turn language: Whisper reports junk codes like "nn" on noise.
                        seg_lang = None
                    elif not seg_lang or str(seg_lang).strip().lower() in ("", "auto", "none"):
                        seg_lang = detect_lang(part)
                    if seg_lang:
                        turn_lang = seg_lang
                    print(
                        f"[STT] Segment done audio_dur={dur:.2f}s "
                        f"infer={time.time()-t0:.2f}s text={part!r} "
                        f"lang={seg_lang}"
                    )
                    continue
                if not text:
                    continue
                tts_lang = os.environ.get("TTS_LANGUAGE") or os.environ.get("TTS_LANG") or lang
                if os.environ.get("ORCHESTRATOR_URL"):
                    # Reuse session inside a follow-up chain so the
                    # orchestrator keeps conversational context.
                    if active_session_id is None:
                        active_session_id = f"voice-{uuid.uuid4().hex[:12]}"
                    session_id = active_session_id
                    # Generation tag: if the user barges in mid-reply, turn_gen
                    # is bumped and this turn's late segments are dropped.
                    req_gen = turn_gen
                    print(f"[ORCH] Sending transcript session={session_id}: {text!r}")
                    is_tts_playing = True
                    tts_play_start = time.time()
                    try:
                        def on_segment(segment, _rg=req_gen):
                            print(f"[ORCH] Response segment: {segment!r}")
                            # Non-blocking: TTS worker synthesizes while the
                            # WS loop keeps receiving LLM tokens.
                            tts_text_queue.put((_rg, segment, tts_lang))
                        send_and_stream(
                            text,
                            session_id,
                            on_segment,
                            n_chunks=TTS_CHUNK_COUNT,
                            language=lang,
                        )
                        # Wait until every queued segment is synthesized, then
                        # until its audio has finished playing.
                        tts_text_queue.join()
                        while not output_queue.empty():
                            time.sleep(0.05)
                        time.sleep(0.3)
                    finally:
                        is_tts_playing = False
                        tts_play_start = None
                        if req_gen != turn_gen:
                            # Superseded by a barge-in: the interrupting turn
                            # owns the conversation now; don't re-arm here.
                            print(
                                f"[barge-in] Turn gen={req_gen} superseded "
                                f"by gen={turn_gen}; skip follow-up"
                            )
                        else:
                            # Open follow-up window: next turn needs no wake-word.
                            ignore_audio_until = time.time() + LISTENING_GRACE_SEC
                            followup_until = time.time() + FOLLOWUP_LISTEN_SEC
                            print(
                                f"[followup] Reply done; listening "
                                f"{FOLLOWUP_LISTEN_SEC:.0f}s for follow-up "
                                f"(session={session_id})"
                            )
                else:
                    tts.speech(
                        text,
                        language=tts_lang,
                        output_queue=output_queue,
                        chunk_size=CHUNK,
                        device_sample_rate=DEVICE_SAMPLE_RATE,
                    )
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[STT][error] {type(e).__name__}: {e}", flush=True)

    def process_tts():
        # Dedicated TTS worker: synthesizes queued segments in order while
        # the WS thread keeps receiving LLM tokens without stalling.
        print(f"[audio][thread] process_tts_tid={get_native_id()}", flush=True)
        while not stop.is_set():
            try:
                item = tts_text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if item is None:
                    continue
                req_gen, text, lang = item
                if req_gen != turn_gen:
                    # Superseded by a barge-in: skip synthesis entirely.
                    print(
                        f"[TTS][worker] Drop stale gen={req_gen} "
                        f"(cur={turn_gen})"
                    )
                    continue
                tts.speech(
                    text,
                    language=lang,
                    output_queue=output_queue,
                    chunk_size=CHUNK,
                    device_sample_rate=DEVICE_SAMPLE_RATE,
                    gen=req_gen,
                )
            except Exception as e:
                print(f"[TTS][worker][error] {type(e).__name__}: {e}", flush=True)
            finally:
                tts_text_queue.task_done()

    t_audio = Thread(target=process_audio, daemon=True)
    t_stt = Thread(target=process_stt, daemon=True)
    t_audio_logs = Thread(target=process_audio_logs, daemon=True)
    t_tts = Thread(target=process_tts, daemon=True)
    t_audio.start()
    t_stt.start()
    t_audio_logs.start()
    t_tts.start()

    startup_tts_lang = os.environ.get("TTS_LANGUAGE") or os.environ.get("TTS_LANG") or "en-us"
    announced_ready = False
    try:
        while not stop.is_set():
            try:
                in_dev, out_dev = select_stream_device()
                with sd.Stream(
                    device=(in_dev, out_dev),
                    samplerate=DEVICE_SAMPLE_RATE,
                    blocksize=CHUNK,
                    channels=CHANNELS,
                    callback=audio_callback,
                    dtype=np.float32,
                    latency=AUDIO_LATENCY,
                ):
                    print(
                        f"[audio] stream opened sample_rate={DEVICE_SAMPLE_RATE} "
                        f"channels={CHANNELS} chunk={CHUNK} latency={AUDIO_LATENCY!r}",
                        flush=True,
                    )
                    record_source = get_current_pulse_record_source()
                    if record_source:
                        print(
                            "[audio] active record source "
                            f"source_output={record_source.get('source_output')} "
                            f"source_index={record_source.get('source_index')} "
                            f"source_name={record_source.get('source_name')!r} "
                            f"application={record_source.get('application_name')!r}",
                            flush=True,
                        )
                    playback_sink = get_current_pulse_playback_sink()
                    if playback_sink:
                        print(
                            "[audio] active playback sink "
                            f"sink_input={playback_sink.get('sink_input')} "
                            f"sink_index={playback_sink.get('sink_index')} "
                            f"sink_name={playback_sink.get('sink_name')!r} "
                            f"application={playback_sink.get('application_name')!r}",
                            flush=True,
                        )
                    if not announced_ready:
                        tts.speech(
                            "FPT Automotive AI is ready",
                            language=startup_tts_lang,
                            output_queue=output_queue,
                            chunk_size=CHUNK,
                            device_sample_rate=DEVICE_SAMPLE_RATE,
                        )
                        announced_ready = True
                    while not stop.is_set():
                        sd.sleep(100)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(
                    f"[audio][error] {type(e).__name__}: {e}; "
                    f"reopening stream in {AUDIO_STREAM_RETRY_SEC:.1f}s",
                    flush=True,
                )
                time.sleep(AUDIO_STREAM_RETRY_SEC)
    finally:
        stop.set()
        t_audio.join(timeout=2)
        t_stt.join(timeout=2)
        t_audio_logs.join(timeout=2)
        t_tts.join(timeout=2)


if __name__ == "__main__":
    run()
