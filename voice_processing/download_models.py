#!/usr/bin/env python3
"""
Download sherpa-onnx models for offline voice processing.

Models downloaded:
  1. KWS (Wake Word) — zipformer gigaspeech (English)
  2. ASR — SenseVoice (zh/en/ja/ko/yue, INT8 quantized)

Usage:
    python3 download_models.py [--model-dir /app/agent_assets/models]

Models are saved to:
    {model_dir}/kws/         ← wake word (sherpa-onnx KeywordSpotter)
    {model_dir}/asr/         ← ASR (sherpa-onnx OfflineRecognizer)

Set SKIP_KWS=1 or SKIP_ASR=1 to skip individual downloads.
"""

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── Model URLs ────────────────────────────────────────────────────────────────
# All sizes are approximate compressed sizes.

KWS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)
KWS_DIRNAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"

ASR_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
)
ASR_DIRNAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

# ── Default keywords file for wake word ───────────────────────────────────────
DEFAULT_KEYWORDS = """\
hey dora @1.8
dola @1.8
"""


def _progress_hook(block_count, block_size, total_size):
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 5) + " " * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {downloaded//1024//1024}MB", end="", flush=True)
    else:
        print(f"\r  {downloaded//1024//1024}MB", end="", flush=True)


def _download_and_extract(url: str, dest_dir: Path, dirname: str) -> Path:
    """Download tar.bz2 archive and extract into dest_dir/dirname."""
    out_path = dest_dir / dirname
    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}", flush=True)
        return out_path

    archive = dest_dir / (dirname + ".tar.bz2")
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, archive, reporthook=_progress_hook)
        print()  # newline after progress bar
    except Exception as e:
        print(f"\n  ERROR downloading {url}: {e}", flush=True)
        if archive.exists():
            archive.unlink()
        raise

    print(f"  Extracting {archive.name}...", flush=True)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(dest_dir)

    archive.unlink()
    print(f"  ✓ Extracted to {out_path}", flush=True)
    return out_path


def _setup_kws(model_base: Path) -> None:
    """Download KWS model and create keywords.txt if not present."""
    kws_dir = model_base / "kws"
    print("\n[KWS] Setting up keyword spotter model...", flush=True)

    src = _download_and_extract(KWS_MODEL_URL, model_base / "_tmp_kws", KWS_DIRNAME)

    # Copy required files into kws/
    kws_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        src_f = src / fname
        if not src_f.exists():
            print(f"  WARNING: {fname} not found in downloaded archive", flush=True)
            continue
        shutil.copy2(src_f, kws_dir / fname)

    # Create keywords.txt with default wake words if not present
    kw_file = kws_dir / "keywords.txt"
    if not kw_file.exists():
        kw_file.write_text(DEFAULT_KEYWORDS)
        print(f"  Created default keywords: {kw_file}", flush=True)
    else:
        print(f"  Keeping existing keywords: {kw_file}", flush=True)

    # Cleanup tmp
    shutil.rmtree(model_base / "_tmp_kws", ignore_errors=True)
    print(f"[KWS] ✓ Ready at {kws_dir}", flush=True)


def _setup_asr(model_base: Path) -> None:
    """Download SenseVoice ASR model."""
    asr_dir = model_base / "asr"
    print("\n[ASR] Setting up SenseVoice ASR model...", flush=True)

    src = _download_and_extract(ASR_MODEL_URL, model_base / "_tmp_asr", ASR_DIRNAME)

    # Copy required files into asr/
    asr_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("model.int8.onnx", "tokens.txt"):
        src_f = src / fname
        if not src_f.exists():
            # Try alternative naming
            candidates = list(src.glob("*.int8.onnx")) + list(src.glob("*.onnx"))
            if candidates and fname.endswith(".onnx"):
                src_f = candidates[0]
                print(f"  Using alternative model file: {src_f.name}", flush=True)
            else:
                print(f"  WARNING: {fname} not found", flush=True)
                continue
        shutil.copy2(src_f, asr_dir / fname)

    # Cleanup tmp
    shutil.rmtree(model_base / "_tmp_asr", ignore_errors=True)
    print(f"[ASR] ✓ Ready at {asr_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Download sherpa-onnx models")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "MODEL_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_assets", "models"),
        ),
        help="Directory to download models into",
    )
    args = parser.parse_args()
    model_base = Path(args.model_dir)
    print(f"Model directory: {model_base}", flush=True)

    skip_kws = os.environ.get("SKIP_KWS", "").strip() in ("1", "true", "yes")
    skip_asr = os.environ.get("SKIP_ASR", "").strip() in ("1", "true", "yes")

    if not skip_kws:
        _setup_kws(model_base)
    else:
        print("[KWS] Skipped (SKIP_KWS=1)", flush=True)

    if not skip_asr:
        _setup_asr(model_base)
    else:
        print("[ASR] Skipped (SKIP_ASR=1)", flush=True)

    print("\n✅ All models ready.", flush=True)


if __name__ == "__main__":
    main()
