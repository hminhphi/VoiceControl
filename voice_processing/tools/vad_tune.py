#!/usr/bin/env python3
"""
vad_tune.py — visualize and tune VAD thresholds / turn-taking for voice_processing.

It runs the SAME Silero VAD + hysteresis as the live pipeline, then prints an
ASCII probability timeline and simulates the segment / turn-end logic used in
main.py, so you can pick thresholds that avoid noise triggers while still
ending a turn promptly.

Examples
--------
# Analyze a recorded WAV (records recommended in the real noisy environment)
python3 tools/vad_tune.py --wav sample.wav

# Record N seconds from the microphone, then analyze
python3 tools/vad_tune.py --record 8

# Sweep a few thresholds to compare false triggers
python3 tools/vad_tune.py --wav sample.wav --sweep

# Try candidate values without editing .env
python3 tools/vad_tune.py --wav sample.wav --detect 0.65 --high 0.65 --low 0.25 \
    --window 4 --segment-silence 1.0 --turn-end 1.0

# Also transcribe each detected segment with the configured STT backend
python3 tools/vad_tune.py --wav sample.wav --stt

Outputs a CSV next to the WAV (or --csv path) for plotting in Excel/gnuplot.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.dirname(HERE)
sys.path.insert(0, VOICE_DIR)

SAMPLE_RATE = 16000
CHUNK = 512  # Silero v4/v5 chunk size at 16 kHz


def _resample(samples: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return samples.astype(np.float32)
    try:
        from resampler import Resampler
        return Resampler(sr, target).resample(samples.astype(np.float32))
    except Exception:
        n = int(len(samples) * target / sr)
        return np.interp(np.linspace(0, len(samples) - 1, n),
                         np.arange(len(samples)), samples).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    import soundfile as sf
    samples, sr = sf.read(path, dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    return _resample(samples, sr, SAMPLE_RATE)


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd
    dev = os.environ.get("MIC_DEVICE_NAME", "default")
    print(f"[vad_tune] Recording {seconds:.0f}s from mic '{dev}' ... speak now", flush=True)
    frames = int(seconds * SAMPLE_RATE)
    rec = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=None)
    sd.wait()
    return rec[:, 0].astype(np.float32)


def probs_from_vad(samples: np.ndarray, high: float, low: float, window: int) -> list[float]:
    """Run the pipeline's VADProcessor and return one probability per 512-sample chunk."""
    os.environ["VAD_THRESHOLD_HIGH"] = str(high)
    os.environ["VAD_THRESHOLD_LOW"] = str(low)
    os.environ["VAD_WINDOW_SIZE"] = str(window)
    from vad import VADProcessor

    vad = VADProcessor(SAMPLE_RATE)
    vad.update_buffer_f32(samples)
    probs: list[float] = []
    while len(vad.buffer) >= CHUNK:
        probs.append(vad.get_prob())
    return probs


def simulate(probs, detect, segment_silence, turn_end, min_segment_sec):
    """Mirror main.py segment / turn-end logic. Returns (events, segments, turns)."""
    dt = CHUNK / SAMPLE_RATE
    high = detect
    events = []           # (t, label)
    segments = []         # (start, end)
    turns = []            # (start, end)
    last_voice_time = None
    speech_start = None
    seg_start = None
    pending = False

    for i, p in enumerate(probs):
        t = i * dt
        if p > detect:
            if last_voice_time is None or speech_start is None:
                speech_start = t
                events.append((t, "speech_start"))
            last_voice_time = t
        if last_voice_time is None:
            continue
        if pending and t >= last_voice_time + turn_end:
            events.append((t, "turn_end"))
            if seg_start is not None:
                segments.append((seg_start, t))
            turns.append((speech_start or t, t))
            seg_start = None
            pending = False
            last_voice_time = None
            speech_start = None
        elif t >= last_voice_time + segment_silence:
            if speech_start is not None and last_voice_time is not None:
                dur = last_voice_time - speech_start
                if dur >= min_segment_sec:
                    events.append((t, "segment"))
                    segments.append((seg_start if seg_start is not None else speech_start, t))
                    seg_start = t
                    pending = True
                    speech_start = t  # next segment starts after this silence
    return events, segments, turns


def render(probs, high, low, segments, turns):
    dt = CHUNK / SAMPLE_RATE
    seg_marks = {int(round(s / dt)) for s, _ in segments}
    turn_marks = {int(round(s / dt)) for s, _ in turns}
    print("\n[vad_tune] legend: '#'>=high  '='>=low  '.'=below   |seg  T=turn_end")
    line = []
    for i, p in enumerate(probs):
        if i in turn_marks:
            ch = "T"
        elif i in seg_marks:
            ch = "|"
        elif p >= high:
            ch = "#"
        elif p >= low:
            ch = "="
        else:
            ch = "."
        line.append(ch)
    width = 60
    for start in range(0, len(line), width):
        t0 = start * dt
        print(f"  {t0:6.2f}s |{''.join(line[start:start + width])}|")
    print()


def suggest(probs):
    if not probs:
        return
    arr = np.asarray(probs)
    noise = float(np.percentile(arr, 20))
    speech = float(np.percentile(arr, 90))
    det = min(0.9, max(0.4, round((noise + speech) / 2, 2)))
    high = min(0.95, max(0.45, round(noise + 0.2, 2)))
    low = min(0.5, max(0.1, round(noise + 0.05, 2)))
    print("[vad_tune] suggested thresholds from this clip:")
    print(f"  noise_p20={noise:.2f}  speech_p90={speech:.2f}")
    print(f"  VAD_DETECT_THRESHOLD={det}  VAD_THRESHOLD_HIGH={high}  VAD_THRESHOLD_LOW={low}")


def transcribe_segments(samples, segments):
    try:
        from stt import STTProcessor
    except Exception as e:
        print(f"[vad_tune] STT unavailable: {e}")
        return
    stt = STTProcessor(SAMPLE_RATE)
    for start, end in segments:
        seg = samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        if len(seg) < int(0.1 * SAMPLE_RATE):
            continue
        audio_i16 = np.clip(seg * 32768, -32768, 32767).astype(np.int16)
        out = stt.transcribe(audio_i16)
        print(f"  [{start:5.2f}-{end:5.2f}s] lang={out.get('language')} text={out.get('text')!r}")


def write_csv(path, probs, high, low):
    dt = CHUNK / SAMPLE_RATE
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "prob", "above_high", "above_low"])
        for i, p in enumerate(probs):
            w.writerow([f"{i * dt:.3f}", f"{p:.4f}", int(p >= high), int(p >= low)])
    print(f"[vad_tune] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="VAD threshold / turn-taking tuner")
    ap.add_argument("--wav", help="input WAV (mono/stereo, any sample rate)")
    ap.add_argument("--record", type=float, help="record N seconds from mic instead of --wav")
    ap.add_argument("--detect", type=float, default=float(os.environ.get("VAD_DETECT_THRESHOLD", "0.55")))
    ap.add_argument("--high", type=float, default=float(os.environ.get("VAD_THRESHOLD_HIGH", "0.55")))
    ap.add_argument("--low", type=float, default=float(os.environ.get("VAD_THRESHOLD_LOW", "0.20")))
    ap.add_argument("--window", type=int, default=int(os.environ.get("VAD_WINDOW_SIZE", "3")))
    ap.add_argument("--segment-silence", type=float, default=float(os.environ.get("SEGMENT_SILENCE", "1.0")))
    ap.add_argument("--turn-end", type=float, default=float(os.environ.get("TURN_END_SILENCE", "1.0")))
    ap.add_argument("--min-segment", type=float, default=float(os.environ.get("MIN_SEGMENT_SEC", "0.20")))
    ap.add_argument("--csv", help="CSV output path")
    ap.add_argument("--sweep", action="store_true", help="compare a few detect thresholds")
    ap.add_argument("--stt", action="store_true", help="transcribe detected segments")
    args = ap.parse_args()

    if args.record and not args.wav:
        samples = record(args.record)
        if not args.csv:
            os.makedirs(os.path.join(VOICE_DIR, "output"), exist_ok=True)
            args.csv = os.path.join(VOICE_DIR, "output", f"vad_tune_{int(time.time())}.csv")
    elif args.wav:
        samples = load_wav(args.wav)
        if not args.csv:
            args.csv = os.path.splitext(args.wav)[0] + "_vad.csv"
    else:
        ap.error("provide --wav PATH or --record SECONDS")

    dur = len(samples) / SAMPLE_RATE
    print(f"[vad_tune] {dur:.2f}s audio @16k; detect={args.detect} high={args.high} "
          f"low={args.low} window={args.window} seg_sil={args.segment_silence} "
          f"turn_end={args.turn_end} min_seg={args.min_segment}")

    probs = probs_from_vad(samples, args.high, args.low, args.window)
    events, segments, turns = simulate(probs, args.detect, args.segment_silence,
                                       args.turn_end, args.min_segment)
    render(probs, args.high, args.low, segments, turns)

    print(f"[vad_tune] segments={len(segments)} turns={len(turns)} "
          f"noise_triggers={sum(1 for _, s, e in [(0, *s) for s in segments] if (e - s) < args.min_segment)}")
    for start, end in segments:
        print(f"  segment {start:5.2f} - {end:5.2f}s  ({end - start:.2f}s)")
    for start, end in turns:
        print(f"  turn    {start:5.2f} - {end:5.2f}s")

    suggest(probs)

    if args.sweep:
        print("\n[vad_tune] sweep (detect -> segments):")
        for d in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
            _, segs, _ = simulate(probs, d, args.segment_silence, args.turn_end, args.min_segment)
            total = sum(e - s for s, e in segs)
            print(f"  detect={d:.2f}  segments={len(segs):2d}  speech={total:5.2f}s")

    write_csv(args.csv, probs, args.high, args.low)

    if args.stt:
        print("\n[vad_tune] STT on detected segments:")
        transcribe_segments(samples, segments)


if __name__ == "__main__":
    main()
