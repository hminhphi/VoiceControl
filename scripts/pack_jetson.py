#!/usr/bin/env python3
"""
pack_jetson.py — build a ready-to-extract Jetson (arm64) deployment zip.

Produces: dist/orchestrator-on-edge-jetson-<YYYYMMDD>.zip
The zip contains a single top-level folder `orchestrator-on-edge/`, so on the
Jetson you can simply `unzip` and `cd orchestrator-on-edge`.

Includes: arm64 config + code + models (GGUF, sherpa, Kokoro, embedding cache,
Jetson aarch64 wheels, arm64 native libs). Ships `.env.example` only.

Excludes: everything x86 (docker-compose.x86.yml, Dockerfile.x86*, *.x86),
dev-only tooling (stubs, run_all_pc.ps1, car_control_ui, frontend), secrets
(.env, .env.x86), venvs, caches-byproducts, build outputs, __pycache__, and
non-arm64 native libs.

Usage:
    python scripts/pack_jetson.py [--out dist] [--name orchestrator-on-edge]
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOP_FILES = [
    ".dockerignore", ".env.example", ".gitignore",
    "LICENSE", "Makefile", "README.md",
    "DEPLOY.md", "DEV_STEP.md", "SYSTEMD_SETUP.md",
    "docker-compose.yml", "Dockerfile.l4t-base",
    "orchestrator-on-edge.service", "pyproject.toml", "uv.lock",
    "run_all.sh", "self_heal.sh", "compress.sh",
    "setup_blue.sh", "setup_blue1.sh", "setup_blue_hfu.sh",
]

TOP_DIRS = [
    "orchestrator", "agents", "shared", "test", "scripts", "docs",
    "voice_processing", "llama-cpp", "cache/orchestrator", "cache/car_manual",
]

# Directory names skipped anywhere in the tree.
EXCLUDE_DIR_NAMES = {
    ".git", ".github", ".vscode", ".idea", ".venv", ".venvs", ".venv_docker",
    "__pycache__", ".pytest_cache", ".tox", ".agents", ".codex",
    "node_modules", "logs", "xet", "stubs", "car_control_ui", "frontend",
}

# Exact relative directories to skip.
EXCLUDE_DIR_REL = {
    "voice_processing/cache", "voice_processing/input_test", "voice_processing/output",
    "voice_processing/agent_assets/models/asr_whisper",
    "voice_processing/agent_assets/models/faster_whisper",
    "voice_processing/agent_assets/models/tts",
    "voice_processing/checkpoints",
    "voice_processing/tools/audio_lab/recordings",
    "voice_processing/torch2trt/build", "voice_processing/torch2trt/dist",
    "voice_processing/whisper_trt/build", "voice_processing/whisper_trt/dist",
    "cache/orchestrator/xet", "cache/car_manual/xet",
}

# File globs skipped anywhere.
EXCLUDE_FILE_GLOBS = [
    "*.pyc", "*.pyo", "*.pyd", "*.log", "*.egg-info",
    "*.part", "*.host-link-backup", ".DS_Store", "Thumbs.db",
]

# x86 / dev-only files skipped anywhere.
EXCLUDE_FILE_NAMES = {
    ".env", ".env.x86", ".env.x86.example",
    "docker-compose.x86.yml", "Dockerfile.x86-base",
    "Dockerfile.x86", "pyproject.x86.toml", "run_all_pc.ps1",
}

# Native libs directory layout: keep only linux/arm64.
LIB_ROOT = "voice_processing/libs/"
LIB_DROP = re.compile(r"/(x64|mac|macos|win|windows)/")

STORE_EXTS = {
    ".gguf", ".onnx", ".whl", ".bin", ".npy", ".npz", ".so", ".dll", ".dylib",
    ".lib", ".a", ".png", ".jpg", ".jpeg", ".wav", ".mp3", ".egg", ".zip",
}


def _skip_dir(rel: str) -> bool:
    name = Path(rel).name
    if name in EXCLUDE_DIR_NAMES:
        return True
    if rel in EXCLUDE_DIR_REL:
        return True
    # cache/voice/torch build dirs living inside an agent/extra folder
    if name == "cache" and rel.startswith("agents/"):
        return True
    return False


def _skip_file(rel: str) -> bool:
    name = Path(rel).name
    if name in EXCLUDE_FILE_NAMES:
        return True
    if name.endswith(".x86"):
        return True
    for pat in EXCLUDE_FILE_GLOBS:
        if fnmatch.fnmatch(name, pat):
            return True
    if rel.startswith(LIB_ROOT) and LIB_DROP.search("/" + rel[len(LIB_ROOT):]):
        return True
    return False


def _iter_files(llm_model_file: str | None = None) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []

    for rel in TOP_FILES:
        p = ROOT / rel
        if p.is_file() and not _skip_file(rel):
            files.append((p, rel))
        elif not p.exists():
            print(f"  [warn] missing top file: {rel}", file=sys.stderr)

    for top in TOP_DIRS:
        base = ROOT / top
        if not base.exists():
            print(f"  [warn] missing dir: {top}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
            rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
            rel_dir = "" if rel_dir == "." else rel_dir
            dirnames[:] = [
                d for d in dirnames
                if not _skip_dir(f"{rel_dir}/{d}" if rel_dir else d)
            ]
            for fn in filenames:
                rel = f"{rel_dir}/{fn}" if rel_dir else fn
                if _skip_file(rel):
                    continue
                # Ship only the GGUF referenced by LLM_MODEL_FILE (avoid bundling
                # every model sitting in llama-cpp/models/).
                if rel.startswith("llama-cpp/models/") and fn.lower().endswith(".gguf"):
                    if llm_model_file and fn != llm_model_file:
                        continue
                p = Path(dirpath) / fn
                if not p.is_file():
                    continue
                files.append((p, rel))
    return files


def _read_llm_model_file() -> str | None:
    """Resolve the model filename to ship: env override, else .env.example."""
    env_val = os.environ.get("LLM_MODEL_FILE")
    if env_val:
        return env_val.strip()
    env_file = ROOT / ".env.example"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("LLM_MODEL_FILE="):
                return line.split("=", 1)[1].strip() or None
    return None


def _compress_type(path: Path) -> int:
    if path.suffix.lower() in STORE_EXTS or path.stat().st_size > 5 * 1024 * 1024:
        return zipfile.ZIP_STORED
    return zipfile.ZIP_DEFLATED


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Jetson arm64 deploy zip")
    parser.add_argument("--out", default="dist", help="output directory (default: dist)")
    parser.add_argument("--name", default="orchestrator-on-edge", help="zip name prefix")
    parser.add_argument("--root-prefix", default="orchestrator-on-edge",
                        help="top-level folder inside the zip")
    parser.add_argument("--llm-model-file", default=None,
                        help="GGUF filename to ship (default: LLM_MODEL_FILE from .env.example)")
    args = parser.parse_args()

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{args.name}-jetson-{date.today():%Y%m%d}.zip"

    llm_model_file = args.llm_model_file or _read_llm_model_file()
    if llm_model_file:
        print(f"[pack_jetson] LLM model: {llm_model_file}")
        if not (ROOT / "llama-cpp" / "models" / llm_model_file).is_file():
            print(f"  [warn] {llm_model_file} not found in llama-cpp/models/ "
                  f"(run scripts/fetch_assets.sh llm)", file=sys.stderr)

    files = _iter_files(llm_model_file)
    total_bytes = sum(p.stat().st_size for p, _ in files)
    print(f"[pack_jetson] {len(files)} files, {total_bytes / 1e9:.2f} GB uncompressed")
    print(f"[pack_jetson] -> {zip_path}")

    t0 = time.time()
    written = 0
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as zf:
        for src, rel in files:
            arc = f"{args.root_prefix}/{rel}"
            zf.write(src, arcname=arc, compress_type=_compress_type(src))
            written += 1
            if written % 50 == 0:
                print(f"  ... {written}/{len(files)}", flush=True)

    size = zip_path.stat().st_size
    print(f"[pack_jetson] done: {written} files, {size / 1e9:.2f} GB on disk, "
          f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
