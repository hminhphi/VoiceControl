#!/usr/bin/env python3
"""
fetch_assets.py — download the large/derived assets that are NOT tracked in git.

Everything this script fetches is listed in docs/ASSETS.md. Assets that are
small and custom (wake-word .onnx, silero_vad.onnx) ARE tracked in git and only
verified here.

Usage (from the repo root):
    python scripts/fetch_assets.py --all
    python scripts/fetch_assets.py llm embed kokoro sherpa wheels
    python scripts/fetch_assets.py verify

Subcommands:
    llm          Qwen3.5-4B Q4_K_M GGUF         -> llama-cpp/models/
    embed        sentence-transformers MiniLM   -> cache/orchestrator/hub/
    kokoro       Kokoro v1.0 ONNX + voices      -> voice_processing/kokoro_tts/
    sherpa       SenseVoice ASR + KWS (delegates to voice_processing/download_models.py)
    asr-whisper  sherpa-onnx Whisper (vi/en/ja) -> .../models/asr_whisper/
    asr-fw       faster-whisper CTranslate2     -> HF cache (WHISPER_MODEL)
    tts-vi       Piper vi_VN (sherpa-onnx)      -> .../models/tts/
    tse          ClearVoice separation/TSE      -> voice_processing/checkpoints/
    speaker      pyannote diarization/embedding -> HF cache (gated, needs HF_TOKEN)
    wheels       Jetson aarch64 wheels          -> voice_processing/wheels/
    silero       nothing to download (tracked)  -> verifies silero_vad.onnx
    verify       check every expected asset exists

Environment:
    HF_TOKEN        HuggingFace token (only needed for gated/private repos)
    WHEELS_INDEX    override the Jetson wheel index (default below)

Idempotent: existing files are skipped. Downloads use a .part temp file and are
atomically renamed, so a partial download never looks complete.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Asset specifications ──────────────────────────────────────────────────────
GGUF_REPO = "unsloth/Qwen3.5-4B-GGUF"
GGUF_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
GGUF_DIR = ROOT / "llama-cpp" / "models"

EMBED_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_CACHE = ROOT / "cache" / "orchestrator" / "hub"

KOKORO_REPO = "mikkoph/kokoro-onnx"  # HF mirror of thewh1teagle/kokoro-onnx releases
KOKORO_DIR = ROOT / "voice_processing" / "kokoro_tts"
KOKORO_FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]

# Multilingual ASR (vi/en/ja + auto language detection) — sherpa-onnx Whisper.
# Size is chosen by STT_WHISPER_MODEL (tiny|base|small|medium), default small.
WHISPER_DIR = ROOT / "voice_processing" / "agent_assets" / "models" / "asr_whisper"


def _whisper_spec():
    size = (os.environ.get("STT_WHISPER_MODEL", "small") or "small").strip().lower()
    repo = f"csukuangfj/sherpa-onnx-whisper-{size}"
    files = [f"{size}-encoder.int8.onnx", f"{size}-decoder.int8.onnx", f"{size}-tokens.txt"]
    return size, repo, files


# faster-whisper (CTranslate2) model dir — GPU multilingual ASR (vi/en/ja).
# The unified config uses the model *name* (WHISPER_MODEL=large-v3), which
# faster-whisper resolves from the HuggingFace cache, so we prefetch into the
# cache (`models--Systran--faster-whisper-<size>`). FW_DIR is kept for optional
# local-dir overrides via FASTER_WHISPER_MODEL.
FW_DIR = ROOT / "voice_processing" / "agent_assets" / "models" / "faster_whisper"
VOICE_CACHE = ROOT / "cache" / "voice_processing"
# Same HF cache the voice container sees (./cache/voice_processing -> /app/cache),
# so prefetched models are shared between host and Docker.
os.environ.setdefault("HF_HOME", str(VOICE_CACHE))
HF_CACHE = Path(os.environ["HF_HOME"]) / "hub"

# Vietnamese TTS — sherpa-onnx Piper vi_VN (VITS).
PIPER_VI_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "tts-models/vits-piper-vi_VN-vais1000-medium.tar.bz2"
)
TTS_DIR = ROOT / "voice_processing" / "agent_assets" / "models" / "tts"

# ClearVoice (speech separation / TSE) — ClearVoice loads from ./checkpoints
# relative to the working dir (voice_processing → /app in the container).
TSE_DIR = ROOT / "voice_processing" / "checkpoints"
CLEARVOICE_MODEL = "MossFormer2_SS_16K"

# pyannote speaker diarization/embedding (models are gated on HuggingFace).
PYANNOTE_REPOS = [
    "pyannote/embedding",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
]

WHISPER_SIZE, _, _ = _whisper_spec()
FW_SIZE = (
    os.environ.get("WHISPER_MODEL")
    or os.environ.get("FASTER_WHISPER_SIZE")
    or "large-v3"
).strip()

WHEELS_DIR = ROOT / "voice_processing" / "wheels"
# All aarch64 wheels come from the NVIDIA Jetson index (JetPack 6 / CUDA 12.6).
# Pinned by sha256 to the known-good r36.4 bundle so a mismatched build can never
# silently replace a working wheel. No pip required — plain HTTPS download.
WHEELS_INDEX = os.environ.get("WHEELS_INDEX", "https://pypi.jetson-ai-lab.io/jp6/cu126")
WHEELS = [
    {
        "file": "torch-2.8.0-cp310-cp310-linux_aarch64.whl",
        "rel": "+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl",
        "sha256": "62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc",
        "size": 225_979_378,
    },
    {
        "file": "torchvision-0.23.0-cp310-cp310-linux_aarch64.whl",
        "rel": "+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl",
        "sha256": "907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1",
        "size": 1_548_174,
    },
    {
        "file": "torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl",
        "rel": "+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl",
        "sha256": "81a775c8af36ac859fb3f4a1b2f662d5fcf284a835b6bb4ed8d0827a6aa9c0b7",
        "size": 2_113_630,
    },
    {
        "file": "onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl",
        "rel": "+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl",
        "sha256": "4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b",
        "size": 87_892_917,
    },
]

ASSETS_ROOT = ROOT / "voice_processing" / "agent_assets" / "models"
SHERPA_DIR = ASSETS_ROOT
DOWNLOAD_MODELS = ROOT / "voice_processing" / "download_models.py"

# sherpa-onnx KWS (wake word) — upstream tar ships epoch-suffixed filenames,
# while the runtime (wake_word.py) expects encoder.onnx/decoder.onnx/joiner.onnx.
KWS_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)
KWS_KEYWORDS = "hey dora @1.8\ndola @1.8\n"

# ── Verification expectations ─────────────────────────────────────────────────
EXPECTED = [
    (GGUF_DIR / GGUF_FILE, 2_740_937_888),
    (KOKORO_DIR / "kokoro-v1.0.onnx", 300_000_000),
    (KOKORO_DIR / "voices-v1.0.bin", 27_553_100),
    (ASSETS_ROOT / "asr" / "model.int8.onnx", 200_000_000),
    (ASSETS_ROOT / "asr" / "tokens.txt", None),
    (ASSETS_ROOT / "kws" / "encoder.onnx", None),
    (ASSETS_ROOT / "silero_vad.onnx", None),          # tracked in git
    (ASSETS_ROOT / "hey_doh_ra.onnx", None),          # tracked in git
    (WHEELS_DIR / "torch-2.8.0-cp310-cp310-linux_aarch64.whl", 225_000_000),
    (WHEELS_DIR / "torchvision-0.23.0-cp310-cp310-linux_aarch64.whl", 1_500_000),
    (WHEELS_DIR / "torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl", 2_000_000),
    (WHEELS_DIR / "onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl", 87_000_000),
    # Multilingual ASR / TTS (x86; optional on Jetson)
    (WHISPER_DIR / f"{WHISPER_SIZE}-encoder.int8.onnx", 50_000_000),
    (WHISPER_DIR / f"{WHISPER_SIZE}-tokens.txt", None),
    (HF_CACHE / f"models--Systran--faster-whisper-{FW_SIZE}", None),
    (TTS_DIR / "vits-piper-vi_VN-vais1000-medium", None),
    # Target Speaker Extraction (ClearVoice separation)
    (TSE_DIR / CLEARVOICE_MODEL / "last_best_checkpoint.pt", 500_000_000),
]


def log(msg: str) -> None:
    print(f"[fetch_assets] {msg}", flush=True)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# ── Generic helpers ───────────────────────────────────────────────────────────
def _urlopen(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "orchestrator-on-edge/fetch_assets"})
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req)


def download_url(url: str, dest: Path, sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256):
        log(f"skip (exists): {_rel(dest)}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"download: {url}")
    with _urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    if sha256 and _sha256(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch for {dest.name}")
    tmp.replace(dest)
    log(f"  -> {_rel(dest)}")
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_download(repo: str, filename: str, dest: Path) -> Path:
    """Download a single file from a public HF repo (falls back to direct URL)."""
    if dest.exists():
        log(f"skip (exists): {_rel(dest)}")
        return dest
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        tmp = hf_hub_download(repo_id=repo, filename=filename, token=os.environ.get("HF_TOKEN"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp, dest)
        log(f"  -> {_rel(dest)} (hf_hub_download)")
        return dest
    except ImportError:
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
        return download_url(url, dest)


# ── Subcommands ───────────────────────────────────────────────────────────────
def fetch_llm() -> None:
    log(f"LLM GGUF: {GGUF_REPO}:{GGUF_FILE}")
    hf_download(GGUF_REPO, GGUF_FILE, GGUF_DIR / GGUF_FILE)


def fetch_embed() -> None:
    log(f"embedding model: {EMBED_REPO}")
    if not (EMBED_CACHE / f"models--{EMBED_REPO.replace('/', '--')}").exists():
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError:
            raise SystemExit(
                "huggingface_hub is required for the embedding model.\n"
                "  pip install 'huggingface_hub>=0.23'"
            )
        snapshot_download(
            repo_id=EMBED_REPO,
            cache_dir=str(EMBED_CACHE),
            token=os.environ.get("HF_TOKEN"),
        )
    log(f"  -> {(EMBED_CACHE / f'models--{EMBED_REPO.replace(chr(47), chr(45)*2)}')}")
    # car_manual mounts cache/orchestrator/hub/... directly; nothing else to do.


def fetch_kokoro() -> None:
    log(f"Kokoro TTS: {KOKORO_REPO}")
    for fn in KOKORO_FILES:
        hf_download(KOKORO_REPO, fn, KOKORO_DIR / fn)


def fetch_sherpa() -> None:
    log("sherpa-onnx ASR (via voice_processing/download_models.py)")
    if not DOWNLOAD_MODELS.exists():
        raise SystemExit(f"missing {DOWNLOAD_MODELS}")

    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env["SKIP_KWS"] = "1"  # KWS is handled below (fixes upstream filename mismatch)
    if (ASSETS_ROOT / "asr" / "model.int8.onnx").exists():
        env["SKIP_ASR"] = "1"
    subprocess.check_call(
        [sys.executable, str(DOWNLOAD_MODELS), "--model-dir", str(SHERPA_DIR)],
        env=env,
    )
    fetch_sherpa_kws()


def _kws_pick(folder: Path, role: str) -> Path | None:
    cands = [f for f in folder.glob(f"{role}*.onnx") if ".int8." not in f.name]
    if not cands:
        cands = sorted(folder.glob(f"{role}*.onnx"))
    return cands[0] if cands else None


def fetch_sherpa_kws() -> None:
    kws_dir = ASSETS_ROOT / "kws"
    if (kws_dir / "encoder.onnx").exists() and (kws_dir / "joiner.onnx").exists():
        log("KWS already present")
        return

    log("KWS wake word model (sherpa-onnx zipformer gigaspeech)")
    kws_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / "kws.tar.bz2"
        download_url(KWS_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(td)

        tokens = next(Path(td).rglob("tokens.txt"), None)
        if tokens is None:
            raise SystemExit("KWS archive did not contain tokens.txt")
        src = tokens.parent
        for role in ("encoder", "decoder", "joiner"):
            picked = _kws_pick(src, role)
            if picked is None:
                raise SystemExit(f"KWS archive missing {role}*.onnx")
            shutil.copy2(picked, kws_dir / f"{role}.onnx")
            log(f"  -> kws/{role}.onnx  (from {picked.name})")
        shutil.copy2(tokens, kws_dir / "tokens.txt")

    kw = kws_dir / "keywords.txt"
    if not kw.exists():
        kw.write_text(KWS_KEYWORDS)
        log(f"  -> kws/keywords.txt (default)")


def fetch_silero() -> None:
    dest = ASSETS_ROOT / "silero_vad.onnx"
    if dest.exists():
        log("silero_vad.onnx present (tracked in git)")
        return
    log("silero_vad.onnx missing — extracting from the silero-vad pip package")
    try:
        import silero_vad  # type: ignore
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "silero-vad==5.1.2"])
        import silero_vad  # type: ignore
    src = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    log(f"  -> {_rel(dest)}")


def fetch_wheels() -> None:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    for w in WHEELS:
        dest = WHEELS_DIR / w["file"]
        if dest.exists() and _sha256(dest) == w["sha256"]:
            log(f"skip (exists): {w['file']}")
            continue
        if dest.exists():
            log(f"replace mismatched: {w['file']}")
            dest.unlink()
        download_url(f"{WHEELS_INDEX}/{w['rel']}", dest, sha256=w["sha256"])


def fetch_asr_whisper() -> None:
    size, repo, files = _whisper_spec()
    log(f"Whisper ASR ({size}, multilingual vi/en/ja): {repo}")
    for fn in files:
        hf_download(repo, fn, WHISPER_DIR / fn)


def fetch_asr_fw() -> None:
    size = FW_SIZE
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub required for asr-fw; pip install huggingface_hub")
    repo = f"Systran/faster-whisper-{size}"
    cached = HF_CACHE / f"models--{repo.replace('/', '--')}"
    if cached.exists():
        log(f"faster-whisper {size} present (HF cache)")
        return
    log(f"faster-whisper {size}: {repo} -> HF cache (WHISPER_MODEL={size})")
    snapshot_download(repo_id=repo)
    log(f"  -> {_rel(cached)}")


def fetch_tse() -> None:
    dest = TSE_DIR / CLEARVOICE_MODEL
    log(f"ClearVoice TSE checkpoint: alibabasglab/{CLEARVOICE_MODEL}")
    for fn in ("last_best_checkpoint.pt", "last_best_checkpoint", "README.md"):
        try:
            hf_download(f"alibabasglab/{CLEARVOICE_MODEL}", fn, dest / fn)
        except Exception as e:
            log(f"  [warn] {fn}: {e}")


def fetch_speaker() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        log("pyannote models are GATED on HuggingFace.")
        log("  1) accept conditions at https://huggingface.co/pyannote/embedding")
        log("  2) run: HF_TOKEN=hf_xxx python scripts/fetch_assets.py speaker")
        return
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        raise SystemExit("huggingface_hub required for speaker; pip install huggingface_hub")
    for repo in PYANNOTE_REPOS:
        log(f"download {repo} -> HF cache")
        snapshot_download(repo_id=repo, token=token)


def fetch_tts_vi() -> None:
    dest = TTS_DIR / "vits-piper-vi_VN-vais1000-medium"
    if dest.exists() and any(dest.glob("*.onnx")):
        log("Piper vi_VN present (tracked? no — downloaded)")
        return
    log("Vietnamese TTS (Piper vi_VN, sherpa-onnx VITS)")
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / "piper_vi.tar.bz2"
        download_url(PIPER_VI_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(TTS_DIR)
    log(f"  -> {_rel(dest)}")


def verify() -> int:
    missing = 0
    log("verifying assets...")
    for path, min_size in EXPECTED:
        if not path.exists():
            print(f"  MISSING  {_rel(path)}")
            missing += 1
        elif min_size and path.stat().st_size < min_size:
            print(f"  TOO SMALL {_rel(path)} ({path.stat().st_size} < {min_size})")
            missing += 1
        else:
            print(f"  ok       {_rel(path)}")
    log(f"{'ALL OK' if not missing else f'{missing} asset(s) missing'} — see docs/ASSETS.md")
    return 1 if missing else 0


SUBCOMMANDS = {
    "llm": fetch_llm,
    "embed": fetch_embed,
    "kokoro": fetch_kokoro,
    "asr-whisper": fetch_asr_whisper,
    "asr-fw": fetch_asr_fw,
    "tts-vi": fetch_tts_vi,
    "tse": fetch_tse,
    "speaker": fetch_speaker,
    "sherpa": fetch_sherpa,
    "silero": fetch_silero,
    "wheels": fetch_wheels,
    "verify": verify,
}
ORDER = ["llm", "embed", "kokoro", "asr-fw", "asr-whisper", "tts-vi", "tse", "speaker", "sherpa", "silero", "wheels"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download orchestrator-on-edge assets")
    parser.add_argument("assets", nargs="*", choices=[*ORDER, "verify"],
                        help="assets to fetch (default: verify)")
    parser.add_argument("--all", action="store_true", help="fetch every downloadable asset")
    args = parser.parse_args()

    if args.all:
        for name in ORDER:
            SUBCOMMANDS[name]()
        return verify()

    targets = args.assets or ["verify"]
    rc = 0
    for name in targets:
        result = SUBCOMMANDS[name]()
        if isinstance(result, int):
            rc = rc or result
    if targets != ["verify"] and not args.all:
        rc = rc or verify()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
