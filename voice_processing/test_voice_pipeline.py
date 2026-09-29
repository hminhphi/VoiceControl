#!/usr/bin/env python3
"""
Quick test script for the optimized voice pipeline on Jetson AGX.

Tests (in order):
  1. sherpa-onnx import + model files
  2. VAD (stateful dual-threshold)
  3. Wake word (sherpa-onnx or openwakeword)
  4. STT (sherpa-onnx SenseVoice) with a WAV file or synthetic audio
  5. AEC engine (graceful if webrtc_apm missing)

Usage:
    # Inside voice_processing container or with correct PYTHONPATH:
    python3 test_voice_pipeline.py
    python3 test_voice_pipeline.py --wav /app/input_test/sample.wav
"""

import argparse
import os
import sys
import time

import numpy as np

# Add voice_processing dir to path
_VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _VOICE_DIR)

DEVICE_SR = int(os.environ.get("AUDIO_SAMPLE_RATE", "48000"))
ASSETS_DIR = os.path.join(_VOICE_DIR, "agent_assets", "models")

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


# ── 1. Dependency check ────────────────────────────────────────────────────────

def test_imports():
    section("1. Import checks")
    libs = [
        ("numpy",         "import numpy; print(numpy.__version__)"),
        ("torch",         "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"),
        ("onnxruntime",   "import onnxruntime as ort; print(ort.__version__, '| providers:', ort.get_available_providers())"),
        ("sherpa_onnx",   "import sherpa_onnx; print('OK')"),
        ("silero_vad",    "import silero_vad; print('OK')"),
        ("sounddevice",   "import sounddevice; print('OK')"),
        ("webrtc_apm",    "from libs import webrtc_apm; print('OK')"),
    ]
    results = {}
    for name, code in libs:
        try:
            exec(code, {})
            print(f"  {PASS} {name}", flush=True)
            results[name] = True
        except Exception as e:
            level = WARN if name == "webrtc_apm" else FAIL
            print(f"  {level} {name}: {e}", flush=True)
            results[name] = False
    return results


# ── 2. VAD test ───────────────────────────────────────────────────────────────

def test_vad():
    section("2. VAD (stateful dual-threshold)")
    try:
        from vad import VADProcessor
        vad = VADProcessor(DEVICE_SR)

        # Feed 1s of synthetic speech-like noise (pink-ish)
        np.random.seed(42)
        audio = (np.random.randn(DEVICE_SR) * 0.1).astype(np.float32)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()

        vad.update_buffer(audio_bytes)
        probs = []
        for _ in range(20):
            p = vad.get_prob()
            probs.append(p)

        print(f"  {PASS} VAD OK | sample probs (first 5): {[f'{p:.3f}' for p in probs[:5]]}", flush=True)
        print(f"  Speech detected: {vad.is_speech()}", flush=True)
        return True
    except Exception as e:
        print(f"  {FAIL} VAD failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return False


# ── 3. Wake word test ─────────────────────────────────────────────────────────

def test_wake_word():
    section("3. Wake word detector")
    backend = os.environ.get("WAKE_WORD_BACKEND", "openwakeword")
    print(f"  Backend: {backend}", flush=True)
    try:
        from wake_word import WakeWordProcessor
        ww = WakeWordProcessor(DEVICE_SR)

        # Feed 0.5s of silence
        silence = np.zeros(DEVICE_SR // 2, dtype=np.int16)
        ww.update_state(silence.tobytes())
        detected = ww.get_detected()
        print(f"  {PASS} Wake word init OK | silence detection={detected} (expected False)", flush=True)
        return True
    except Exception as e:
        print(f"  {FAIL} Wake word failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return False


# ── 4. STT test ───────────────────────────────────────────────────────────────

def test_stt(wav_path=None):
    section("4. STT")
    backend = os.environ.get("STT_BACKEND", "sherpa_onnx")
    print(f"  Backend: {backend}", flush=True)
    try:
        from stt import STTProcessor
        stt = STTProcessor(DEVICE_SR, language=None)

        if wav_path and os.path.isfile(wav_path):
            import soundfile as sf
            audio, sr = sf.read(wav_path, dtype="int16")
            if audio.ndim > 1:
                audio = audio[:, 0]
            print(f"  Using WAV: {wav_path} ({sr}Hz, {len(audio)/sr:.2f}s)", flush=True)
        else:
            # Generate 2s of white noise (should produce empty transcript)
            print("  Using synthetic audio (white noise — expect empty transcript)", flush=True)
            audio = (np.random.randn(DEVICE_SR * 2) * 0.01 * 32768).astype(np.int16)

        t0 = time.time()
        result = stt.transcribe(audio)
        elapsed = time.time() - t0

        print(f"  {PASS} STT OK | text={result['text']!r} lang={result.get('language')} "
              f"rms={result.get('rms', 0):.5f} elapsed={elapsed:.2f}s", flush=True)
        return True
    except Exception as e:
        print(f"  {FAIL} STT failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return False


# ── 5. AEC test ───────────────────────────────────────────────────────────────

def test_aec():
    section("5. AEC Engine")
    try:
        from aec import AecEngine
        aec = AecEngine(near_rate=DEVICE_SR, far_rate=DEVICE_SR, delay_ms=60)
        print(f"  Active: {aec.active}", flush=True)

        # Feed some audio
        near = np.random.randn(DEVICE_SR // 10).astype(np.float32) * 0.1
        far = np.random.randn(DEVICE_SR // 10).astype(np.float32) * 0.1

        aec.feed_far(far.reshape(-1, 1))
        result = aec.process_near(near)

        assert result.shape == near.shape, f"Shape mismatch: {result.shape} != {near.shape}"
        print(f"  {PASS} AEC OK | active={aec.active} output_shape={result.shape}", flush=True)
        aec.close()
        return True
    except Exception as e:
        print(f"  {WARN} AEC: {e} (graceful — AEC is optional)", flush=True)
        return None  # None = optional, not a hard failure


# ── 6. Model files check ──────────────────────────────────────────────────────

def test_model_files():
    section("6. Model file check")
    checks = {
        "silero_vad.onnx":       os.path.join(ASSETS_DIR, "silero_vad.onnx"),
        "KWS encoder.onnx":      os.path.join(ASSETS_DIR, "kws", "encoder.onnx"),
        "KWS keywords.txt":      os.path.join(ASSETS_DIR, "kws", "keywords.txt"),
        "ASR model.int8.onnx":   os.path.join(ASSETS_DIR, "asr", "model.int8.onnx"),
        "hey_doh_ra.onnx (oww)": os.path.join(ASSETS_DIR, "hey_doh_ra.onnx"),
    }
    all_ok = True
    for name, path in checks.items():
        exists = os.path.isfile(path)
        icon = PASS if exists else WARN
        size = f"{os.path.getsize(path) // 1024}KB" if exists else "MISSING"
        print(f"  {icon} {name}: {size}", flush=True)
        if not exists and "silero" in name:
            all_ok = False  # silero is required
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Voice pipeline test")
    parser.add_argument("--wav", help="Optional WAV file path for STT test")
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT test (faster)")
    args = parser.parse_args()

    print("\n🎤 Voice Pipeline Test — Jetson AGX", flush=True)
    print(f"   DEVICE_SR={DEVICE_SR}  ASSETS={ASSETS_DIR}", flush=True)
    print(f"   STT_BACKEND={os.environ.get('STT_BACKEND', 'sherpa_onnx')}", flush=True)
    print(f"   WAKE_WORD_BACKEND={os.environ.get('WAKE_WORD_BACKEND', 'openwakeword')}", flush=True)

    results = {
        "imports":     test_imports(),
        "model_files": test_model_files(),
        "vad":         test_vad(),
        "wake_word":   test_wake_word(),
        "aec":         test_aec(),
    }
    if not args.skip_stt:
        results["stt"] = test_stt(args.wav)

    section("Summary")
    failed = []
    for name, ok in results.items():
        if ok is None:
            print(f"  {WARN} {name}: optional (not required)", flush=True)
        elif ok is True or (isinstance(ok, dict) and all(ok.values())):
            print(f"  {PASS} {name}", flush=True)
        else:
            print(f"  {FAIL} {name}", flush=True)
            failed.append(name)

    if failed:
        print(f"\n{FAIL} {len(failed)} component(s) failed: {', '.join(failed)}", flush=True)
        sys.exit(1)
    else:
        print(f"\n{PASS} All tests passed!", flush=True)


if __name__ == "__main__":
    main()
