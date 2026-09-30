#!/usr/bin/env python3
"""
asr_benchmark.py — end-to-end ASR benchmark for the voice pipeline.

Datasets (auto-downloaded from HuggingFace):
  * google/fleurs         clean read speech (vi/ja/en) — multilingual WER/CER
  * corypaik/musan        noise subcorpus (technical/ambient) — SNR mixing

Modes per utterance:
  clean | noisy(SNR dB) | overlap(2 speakers) | overlap+TSE(ClearVoice separation)

It runs the SAME backend as the pipeline (faster-whisper) and reports:
  * WER (en) / CER (vi, ja)
  * language-detection accuracy (auto)
  * TSE: WER/CER after target-speaker extraction

Usage:
  python scripts/asr_benchmark.py --langs vi ja en --limit 40 --model large-v3 --device cuda
  python scripts/asr_benchmark.py --langs en --limit 100 --model small --device cpu
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VP = ROOT / "voice_processing"
sys.path.insert(0, str(VP))

# ctranslate2 (faster-whisper) needs cuDNN/cuBLAS on PATH (Windows) — torch ships them.
try:
    import torch
    os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402

LANGS = {"en": "en_us", "vi": "vi_vn", "ja": "ja_jp"}
SR = 16000
_PUNCT = re.compile(r"[^\w\s\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7a3]", re.UNICODE)


def norm(text: str) -> str:
    return _PUNCT.sub(" ", (text or "").lower()).strip()


def metrics(ref: str, hyp: str, lang: str) -> float:
    import jiwer
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 1.0
    if lang == "en":
        return float(jiwer.wer(ref, hyp))
    return float(jiwer.cer(ref, hyp))


def resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    n = int(len(x) * target / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = len(speech)
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / max(len(noise), 1))))
    noise = noise[:n]
    sp = float(np.sqrt(np.mean(speech ** 2)) + 1e-9)
    npow = float(np.sqrt(np.mean(noise ** 2)) + 1e-9)
    target = sp / (10 ** (snr_db / 20.0))
    return (speech + noise * (target / npow)).astype(np.float32)


def _decode(audio_field) -> tuple[np.ndarray, int]:
    """Decode an HF audio dict (decode=False) via soundfile — no torchcodec needed."""
    import io
    import soundfile as sf
    if isinstance(audio_field, dict) and audio_field.get("bytes"):
        x, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
    elif isinstance(audio_field, dict) and audio_field.get("path"):
        x, sr = sf.read(audio_field["path"], dtype="float32")
    else:  # already decoded (array/sampling_rate)
        return np.asarray(audio_field["array"], dtype=np.float32), int(audio_field["sampling_rate"])
    if x.ndim > 1:
        x = x[:, 0]
    return x.astype(np.float32), int(sr)


def load_fleurs(lang_code: str, limit: int):
    from datasets import Audio, load_dataset
    ds = load_dataset("google/fleurs", lang_code, split="test").cast_column("audio", Audio(decode=False))
    out = []
    for ex in ds:
        x, sr = _decode(ex["audio"])
        out.append((resample(x, sr), ex.get("transcription") or ex.get("raw_transcription") or ""))
        if len(out) >= limit:
            break
    return out


def load_noise_pool(seconds: int = 120):
    from datasets import Audio, load_dataset
    ds = load_dataset("corypaik/musan", "noise", split="train").cast_column("audio", Audio(decode=False))
    pool, total = [], 0
    for ex in ds:
        x, sr = _decode(ex["audio"])
        x = resample(x, sr)
        pool.append(x)
        total += len(x)
        if total >= seconds * SR:
            break
    return np.concatenate(pool) if pool else np.zeros(seconds * SR, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "vi", "ja"])
    ap.add_argument("--limit", type=int, default=40, help="utterances per language per mode")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute", default=None, help="float16/int8/default(auto)")
    ap.add_argument("--snr", type=float, nargs="+", default=[10.0, 0.0])
    ap.add_argument("--tse", action="store_true", help="also test overlap + TSE")
    ap.add_argument("--out", default=None, help="write markdown report")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    compute = args.compute or ("float16" if args.device == "cuda" else "int8")
    print(f"[bench] faster-whisper model={args.model} device={args.device} compute={compute}", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=compute)

    def asr(wav: np.ndarray, lang: str | None):
        segs, info = model.transcribe(wav, language=lang, beam_size=5, vad_filter=True,
                                      condition_on_previous_text=False, temperature=0.0)
        return "".join(s.text for s in segs).strip(), (info.language or "")

    extractor = None
    if args.tse:
        try:
            from tse import TargetSpeakerExtractor
            extractor = TargetSpeakerExtractor()
        except Exception as e:
            print(f"[bench] TSE unavailable: {e}", flush=True)

    noise = load_noise_pool()
    rows = []
    for lang in args.langs:
        code = LANGS[lang]
        data = load_fleurs(code, args.limit)
        print(f"[bench] {lang} ({code}): {len(data)} utterances", flush=True)

        # ── clean ──
        wer, n = 0.0, 0
        lang_ok = 0
        t0 = time.time()
        for wav, ref in data:
            hyp, det = asr(wav, lang)
            wer += metrics(ref, hyp, lang); n += 1
            lang_ok += int(det == lang)
        rows.append((lang, "clean", n, wer / max(n, 1), lang_ok / max(n, 1), time.time() - t0))
        print(f"[bench] {lang} clean: {rows[-1][3]:.3f} ({rows[-1][5]:.1f}s)", flush=True)

        # ── noisy at SNR levels ──
        for snr in args.snr:
            wer, n, t0 = 0.0, 0, time.time()
            for wav, ref in data:
                noisy = mix_at_snr(wav, noise, snr)
                hyp, _ = asr(noisy, lang)
                wer += metrics(ref, hyp, lang); n += 1
            rows.append((lang, f"noisy_{snr:g}dB", n, wer / max(n, 1), float("nan"), time.time() - t0))
            print(f"[bench] {lang} noisy {snr:g}dB: {rows[-1][3]:.3f}", flush=True)

        # ── overlap (speaker A first, B joins +1s) ──
        pairs = []
        for i in range(0, min(len(data) - 1, args.limit)):
            pairs.append((data[i], data[i + 1]))
        wer, n, t0 = 0.0, 0, time.time()
        for (wa, ra), (wb, _rb) in pairs:
            off = int(1.0 * SR)
            wb_p = np.concatenate([np.zeros(off, np.float32), wb])
            m = min(len(wa), len(wb_p))
            mixed = (0.5 * wa[:m] + 0.5 * wb_p[:m]).astype(np.float32)
            hyp, _ = asr(mixed, lang)
            wer += metrics(ra, hyp, lang); n += 1
        rows.append((lang, "overlap", n, wer / max(n, 1), float("nan"), time.time() - t0))
        print(f"[bench] {lang} overlap: {rows[-1][3]:.3f}", flush=True)

        # ── overlap + TSE ──
        if extractor is not None:
            wer, n, t0 = 0.0, 0, time.time()
            for (wa, ra), (wb, _rb) in pairs:
                off = int(1.0 * SR)
                wb_p = np.concatenate([np.zeros(off, np.float32), wb])
                m = min(len(wa), len(wb_p))
                mixed = (0.5 * wa[:m] + 0.5 * wb_p[:m]).astype(np.float32)
                try:
                    from speaker import get_analyzer
                    target = extractor.extract_target(mixed, None)
                except Exception:
                    target = mixed
                hyp, _ = asr(target, lang)
                wer += metrics(ra, hyp, lang); n += 1
            rows.append((lang, "overlap+TSE", n, wer / max(n, 1), float("nan"), time.time() - t0))
            print(f"[bench] {lang} overlap+TSE: {rows[-1][3]:.3f}", flush=True)

    # ── report ──
    lines = ["| lang | mode | n | WER/CER | lang-det | time (s) |", "|---|---|---|---|---|---|"]
    for lang, mode, n, err, det, dt in rows:
        det_s = "—" if det != det or det == float("nan") else f"{det*100:.0f}%"
        lines.append(f"| {lang} | {mode} | {n} | {err:.3f} | {det_s} | {dt:.1f} |")
    report = "\n".join(lines)
    print("\n" + report)
    if args.out:
        Path(args.out).write_text(f"# ASR benchmark ({args.model}, {args.device})\n\n{report}\n", encoding="utf-8")
        print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
