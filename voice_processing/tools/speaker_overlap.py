#!/usr/bin/env python3
"""
speaker_overlap.py — PoC: detect overlapped speech (multiple speakers talking at
once) and print a diarization timeline, based on the repos surveyed in
docs/SPEAKER_OVERLAP.md.

Backends (auto-detected, opt-in):
  pyannote    pyannote.audio speaker diarization (model gated on HF: needs token
              and accepting the model conditions). Best accuracy.
  3dspeaker   modelscope/3D-Speaker diarization with --include_overlap
              (Apache-2.0). Point THREEDSPEAKER_DIR to a clone.
  vad         dependency-light fallback: uses the project's Silero VAD to show a
              speech/silence timeline (no speaker labels / no real overlap).

Examples:
  python tools/speaker_overlap.py --wav mixed.wav
  python tools/speaker_overlap.py --wav mixed.wav --backend pyannote --hf-token hf_xxx
  THREEDSPEAKER_DIR=~/3D-Speaker python tools/speaker_overlap.py --wav mixed.wav --backend 3dspeaker

Target-speaker extraction (extract the person talking to the car) is handled by
ClearerVoice-Studio (Apache-2.0) — see docs/SPEAKER_OVERLAP.md §4/§5.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICE_DIR = HERE.parent
sys.path.insert(0, str(VOICE_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SAMPLE_RATE = 16000


def _load_wav(path: str):
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != SAMPLE_RATE:
        n = int(len(x) * SAMPLE_RATE / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
    return x, SAMPLE_RATE


# ── pyannote backend ────────────────────────────────────────────────────────
def diarize_pyannote(wav: str, token: str | None):
    from pyannote.audio import Pipeline
    name = os.environ.get("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1")
    print(f"[overlap] pyannote pipeline: {name}")
    pipe = Pipeline.from_pretrained(name, token=token)
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except Exception:
        pass
    out = pipe(wav)
    diar = getattr(out, "speaker_diarization", out)  # newer returns wrapper
    turns = [(float(t.start), float(t.end), str(spk))
             for t, _, spk in diar.itertracks(yield_label=True)]
    return sorted(turns)


# ── 3D-Speaker backend ──────────────────────────────────────────────────────
def diarize_3dspeaker(wav: str, hf_token: str | None):
    root = os.environ.get("THREEDSPEAKER_DIR", "")
    if not root or not Path(root).is_dir():
        raise RuntimeError(
            "3D-Speaker not configured. Clone it and set THREEDSPEAKER_DIR, e.g.\n"
            "  git clone https://github.com/modelscope/3D-Speaker\n"
            "  export THREEDSPEAKER_DIR=$PWD/3D-Speaker\n"
            "then: python speakerlab/bin/infer_diarization.py --wav <wav> --include_overlap"
        )
    out_dir = HERE / "out_overlap"
    out_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, "speakerlab/bin/infer_diarization.py",
           "--wav", wav, "--out_dir", str(out_dir)]
    if hf_token:
        cmd += ["--include_overlap", "--hf_access_token", hf_token]
    print(f"[overlap] running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=root)
    return _parse_rttm(out_dir)


def _parse_rttm(folder: Path):
    turns = []
    for f in folder.glob("*.rttm"):
        for line in f.read_text(errors="ignore").splitlines():
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                start, dur, spk = float(p[3]), float(p[4]), p[7]
                turns.append((start, start + dur, spk))
    return sorted(turns)


# ── VAD fallback ────────────────────────────────────────────────────────────
def vad_timeline(wav: str):
    import numpy as np
    x, sr = _load_wav(wav)
    from vad import VADProcessor, MODEL_CHUNK
    v = VADProcessor(sr)
    v.update_buffer_f32(x)
    probs = []
    while len(v.buffer) >= MODEL_CHUNK:
        probs.append(v.get_prob())
    thr = float(os.environ.get("VAD_DETECT_THRESHOLD", "0.55"))
    dt = MODEL_CHUNK / sr
    print(f"[overlap] (fallback) VAD timeline @ {dt*1000:.0f}ms, threshold={thr}")
    line = "".join("#" if p >= thr else ("=" if p > 0.2 else ".") for p in probs)
    w = 80
    for i in range(0, len(line), w):
        print(f"  {i*dt:6.2f}s |{line[i:i+w]}|")
    print("  (# speech, = mid, . silence) — không phân biệt người nói / overlap.")


# ── Overlap computation + report ────────────────────────────────────────────
def find_overlaps(turns):
    """Return merged intervals where >=2 distinct speakers are active."""
    events = []
    for s, e, spk in turns:
        events.append((s, +1, spk))
        events.append((e, -1, spk))
    events.sort(key=lambda x: (x[0], x[1]))
    active = {}
    overlaps = []
    open_start = None
    for t, delta, spk in events:
        n_before = sum(1 for c in active.values() if c > 0)
        if delta > 0:
            active[spk] = active.get(spk, 0) + 1
        else:
            active[spk] = active.get(spk, 0) - 1
        n_after = sum(1 for c in active.values() if c > 0)
        if n_before < 2 <= n_after and open_start is None:
            open_start = t
        elif n_before >= 2 > n_after and open_start is not None:
            overlaps.append((open_start, t))
            open_start = None
    if open_start is not None:
        overlaps.append((open_start, events[-1][0]))
    return overlaps


def report(turns, overlaps, total_dur):
    print("\n[overlap] speaker turns:")
    for s, e, spk in turns:
        print(f"  {s:7.2f} - {e:7.2f}s  {spk}")
    ov = sum(e - s for s, e in overlaps)
    print(f"\n[overlap] overlap regions: {len(overlaps)}  "
          f"total={ov:.2f}s ({ov/max(total_dur,1e-6)*100:.1f}% of audio)")
    for s, e in overlaps:
        print(f"  OVERLAP {s:7.2f} - {e:7.2f}s  ({e-s:.2f}s)")


def write_rttm(turns, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s, e, spk in turns:
            f.write(f"SPEAKER audio 1 {s:.3f} {e-s:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    print(f"[overlap] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Overlapped-speech / diarization PoC")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--backend", choices=["auto", "pyannote", "3dspeaker", "vad"], default="auto")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        try:
            import pyannote.audio  # noqa: F401
            backend = "pyannote"
        except Exception:
            backend = "vad"
        print(f"[overlap] auto backend -> {backend}")

    if backend == "vad":
        print("[overlap] pyannote chưa cài. Cài để diarization thật:\n"
              "  pip install pyannote.audio   (cần ffmpeg + accept model ở HF)\n"
              "hoặc dùng 3D-Speaker (Apache-2.0) với --backend 3dspeaker.")
        vad_timeline(args.wav)
        return

    if backend == "pyannote":
        import numpy as np
        x, _ = _load_wav(args.wav)
        turns = diarize_pyannote(args.wav, args.hf_token)
    else:  # 3dspeaker
        turns = diarize_3dspeaker(args.wav, args.hf_token)

    if not turns:
        print("[overlap] no speaker turns found")
        return
    total = max(e for _, e, _ in turns)
    overlaps = find_overlaps(turns)
    report(turns, overlaps, total)
    out = Path(args.out) if args.out else Path(args.wav).with_suffix(".rttm")
    write_rttm(turns, out)


if __name__ == "__main__":
    main()
