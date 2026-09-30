#!/usr/bin/env python3
"""
audio_lab — record a clip and compare it BEFORE vs AFTER the voice_processing
preprocessing chain (WebRTC APM: AEC/NS/HPF/transient/AGC + optional denoise).

Run:
    pip install -r tools/audio_lab/requirements.txt
    python tools/audio_lab/app.py            # then open http://localhost:8020
    # options: --host 0.0.0.0 --port 8020 --aec-far-note

The UI records from the default mic (or lets you upload a WAV), applies the
preprocessing chain, and shows waveform, spectrogram and dBFS levels side by
side, with audio players for A/B listening.

Note: true echo cancellation (AEC) needs a simultaneous far-end (speaker)
signal. In this offline lab there is no playback reference, so AEC has nothing
to cancel; NS / HPF / transient / AGC / denoise still apply. Run the lab while
the vehicle TTS is playing (and tick "use speaker as far-end") to exercise AEC.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

HERE = Path(__file__).resolve().parent
VOICE_DIR = HERE.parents[1]            # .../voice_processing
sys.path.insert(0, str(VOICE_DIR))

REC_DIR = HERE / "recordings"
REC_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))

app = FastAPI(title="Audio Lab — preprocessing A/B")

# ── Optional denoise backends ───────────────────────────────────────────────
try:
    import noisereduce as nr
    HAS_NR = True
except Exception:
    nr = None
    HAS_NR = False

try:
    import multipart  # noqa: F401  (needed by FastAPI UploadFile)
    HAS_MULTIPART = True
except Exception:
    HAS_MULTIPART = False


# ── WAV IO ──────────────────────────────────────────────────────────────────
def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    try:
        from resampler import Resampler
        return Resampler(sr, target).resample(x.astype(np.float32))
    except Exception:
        n = int(len(x) * target / sr)
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def load_wav(path: Path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return x.astype(np.float32), int(sr)


def save_wav(path: Path, x: np.ndarray, sr: int):
    import soundfile as sf
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x, sr, subtype="PCM_16")


# ── Analysis / visualization ────────────────────────────────────────────────
def _dbfs(v: float) -> float:
    return float(max(20.0 * np.log10(v), -120.0)) if v > 0 else -120.0


def _waveform_envelope(x: np.ndarray, points: int = 900):
    """Return (mins, maxs) envelope downsampled to `points` buckets, plus rms arr."""
    n = len(x)
    if n == 0:
        return [], [], []
    idx = np.linspace(0, n, points + 1, dtype=int)
    mins, maxs, rms = [], [], []
    for i in range(points):
        seg = x[idx[i]:idx[i + 1]]
        if len(seg) == 0:
            mins.append(0.0); maxs.append(0.0); rms.append(0.0); continue
        mins.append(float(seg.min()))
        maxs.append(float(seg.max()))
        rms.append(float(np.sqrt(np.mean(seg * seg))))
    return mins, maxs, rms


def _spectrogram_db(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160,
                    nbins: int = 96, ntime: int = 220):
    """Simple STFT magnitude (dB), downsampled to nbins x ntime."""
    if len(x) < n_fft:
        return {"nbins": 0, "ntime": 0, "lo_db": -120.0, "hi_db": 0.0, "data": [], "max_freq": sr / 2}
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(x) - n_fft) // hop
    # cap frames to keep it fast
    if n_frames > 4000:
        hop = max(hop, (len(x) - n_fft) // 4000)
        n_frames = 1 + (len(x) - n_fft) // hop
    mag = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft] * window
        mag[i] = np.abs(np.fft.rfft(seg))
    db = 20.0 * np.log10(mag + 1e-6)

    # downsample time
    t_idx = np.linspace(0, n_frames, ntime + 1, dtype=int)
    db_t = np.stack([db[t_idx[i]:max(t_idx[i] + 1, t_idx[i + 1])].mean(axis=0) for i in range(ntime)])
    # downsample freq
    f_bins = db_t.shape[1]
    nbins = min(nbins, f_bins)
    f_idx = np.linspace(0, f_bins, nbins + 1, dtype=int)
    db_ft = np.stack([db_t[:, f_idx[j]:max(f_idx[j] + 1, f_idx[j + 1])].mean(axis=1) for j in range(nbins)])

    lo = float(np.percentile(db_ft, 5))
    hi = float(np.percentile(db_ft, 99.5))
    norm = np.clip((db_ft - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return {
        "nbins": int(nbins), "ntime": int(ntime),
        "lo_db": round(lo, 1), "hi_db": round(hi, 1),
        "max_freq": int(sr / 2),
        "data": [round(float(v), 3) for v in norm.reshape(-1)],
    }


def analyze(path: Path):
    x, sr = load_wav(path)
    x16 = _resample(x, sr, 16000)
    mins, maxs, rms = _waveform_envelope(x16)
    peak = float(np.max(np.abs(x16))) if len(x16) else 0.0
    return {
        "name": path.name,
        "sr": sr,
        "duration": round(len(x16) / 16000.0, 3),
        "rms_dbfs": round(_dbfs(float(np.sqrt(np.mean(x16 ** 2))) if len(x16) else 0.0), 1),
        "peak_dbfs": round(_dbfs(peak), 1),
        "wave_min": [round(v, 4) for v in mins],
        "wave_max": [round(v, 4) for v in maxs],
        "wave_rms": [round(v, 4) for v in rms],
        "spec": _spectrogram_db(x16, 16000),
    }


# ── Preprocessing chain ─────────────────────────────────────────────────────
def process_file(src: Path, opts: dict) -> Path:
    """Apply the voice_processing chain, save processed WAV, return its path."""
    from aec import AecEngine

    x, sr = load_wav(src)

    # The APM reads its config from env at construction time.
    os.environ["AEC_ENABLED"] = "1" if opts.get("aec", True) else "0"
    os.environ["AEC_NOISE_SUPPRESS"] = "1" if opts.get("ns", True) else "0"
    os.environ["AEC_NS_LEVEL"] = str(opts.get("ns_level", "high"))
    os.environ["AEC_NS_LINEAR"] = "1" if opts.get("ns_linear") else "0"
    os.environ["AEC_HPF_FULL_BAND"] = "1" if opts.get("hpf", True) else "0"
    os.environ["AEC_TRANSIENT_SUPPRESS"] = "1" if opts.get("transient", True) else "0"
    os.environ["AEC_AGC_ENABLED"] = "1" if opts.get("agc2", True) else "0"
    os.environ["AEC_AGC_MAX_GAIN_DB"] = str(opts.get("agc2_max_gain", 30.0))
    os.environ["AEC_AGC_MAX_NOISE_DBFS"] = str(opts.get("agc2_max_noise", -50.0))
    os.environ["AEC_AGC1_ENABLED"] = "1" if opts.get("agc1") else "0"
    os.environ["AEC_AGC1_TARGET_DBFS"] = str(opts.get("agc1_target", -3.0))
    os.environ["AEC_AGC1_LIMITER"] = "1"
    os.environ["AEC_PRE_GAIN"] = str(opts.get("pre_gain", 1.0))

    out = x
    if opts.get("aec", True):
        far = None
        if opts.get("far_file"):
            try:
                far, _ = load_wav(REC_DIR / str(opts["far_file"]))
            except Exception:
                far = None
        aec = AecEngine(near_rate=sr, far_rate=sr,
                        delay_ms=int(opts.get("delay_ms", 60)),
                        enable_preprocess=bool(opts.get("ns", True)))
        try:
            if far is not None and len(far):
                aec.feed_far(far)
            out = aec.process_near(x)
        finally:
            aec.close()

    # Optional spectral-gating denoise (noisereduce), off by default.
    if opts.get("denoise") and HAS_NR:
        try:
            out = nr.reduce_noise(y=out, sr=sr, stationary=bool(opts.get("denoise_stationary", False)))
        except Exception as e:
            print(f"[audio_lab] denoise failed: {e}")

    dst = src.with_name(src.stem + f"_proc_{uuid.uuid4().hex[:6]}.wav")
    save_wav(dst, out, sr)
    return dst


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(
        request=request, name="index.html",
        context={"has_nr": HAS_NR, "voice_dir": str(VOICE_DIR)},
    )


@app.get("/api/capabilities")
async def capabilities():
    import sounddevice as sd
    devs = []
    try:
        for d in sd.query_devices():
            if d.get("max_input_channels", 0) > 0:
                devs.append(d["name"])
    except Exception:
        pass
    return {"denoise": HAS_NR, "upload": HAS_MULTIPART, "input_devices": devs}


@app.post("/api/record")
async def record(body: dict):
    import sounddevice as sd
    import soundfile as sf
    seconds = float(body.get("seconds", 4.0))
    sr = int(body.get("sr", 16000))
    dev = body.get("device") or None
    print(f"[audio_lab] recording {seconds}s @ {sr}Hz", flush=True)
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=dev)
    sd.wait()
    fname = f"rec_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    save_wav(REC_DIR / fname, rec[:, 0], sr)
    return {"file": fname}


if HAS_MULTIPART:
    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        fname = f"up_{uuid.uuid4().hex[:8]}{suffix}"
        data = await file.read()
        (REC_DIR / fname).write_bytes(data)
        try:
            load_wav(REC_DIR / fname)
        except Exception as e:
            return JSONResponse({"error": f"not a readable wav: {e}"}, status_code=400)
        return {"file": fname}


@app.get("/api/analyze")
async def analyze_endpoint(file: str):
    p = REC_DIR / file
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return analyze(p)


@app.post("/api/process")
async def process(body: dict):
    src = REC_DIR / str(body.get("file", ""))
    if not src.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    src = _resample_to(src, int(body.get("sr", 16000))) if body.get("mono16") else src
    out = process_file(src, body.get("opts", {}))
    return {"processed_file": out.name}


def _resample_to(src: Path, sr: int) -> Path:
    x, cur = load_wav(src)
    if cur == sr:
        return src
    dst = src.with_name(src.stem + f"_{sr}.wav")
    save_wav(dst, _resample(x, cur, sr), sr)
    return dst


@app.get("/media/{name}")
async def media(name: str):
    p = REC_DIR / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="audio/wav", filename=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    import uvicorn
    print(f"[audio_lab] http://{args.host}:{args.port}  (denoise={HAS_NR})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
