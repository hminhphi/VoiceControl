# Assets & Model Sources

Large/derived assets are **not tracked in git**. Small custom assets **are
tracked**. Use `scripts/fetch_assets.*` to fetch everything, then
`scripts/pack_jetson.*` to build the arm64 deploy zip.

```bash
# Linux / macOS / Jetson
scripts/fetch_assets.sh --all
# Windows (PowerShell)
.\scripts\fetch_assets.ps1 --all
```

Set `HF_TOKEN` in the environment only for gated/private repos (none of the
public defaults need it).

## Source map

| Asset | Tracked in git? | Destination | Source | Fetcher |
|-------|-----------------|-------------|--------|---------|
| GGUF `Qwen3.5-4B-Q4_K_M.gguf` | no (2.7 GB) | `llama-cpp/models/` | HF `unsloth/Qwen3.5-4B-GGUF` | `fetch_assets llm` |
| Embedding `paraphrase-multilingual-MiniLM-L12-v2` | no (470 MB) | `cache/orchestrator/hub/` | HF `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `fetch_assets embed` |
| Kokoro TTS `kokoro-v1.0.onnx`, `voices-v1.0.bin` | no (345 MB) | `voice_processing/kokoro_tts/` | HF `mikkoph/kokoro-onnx` (mirror) | `fetch_assets kokoro` |
| sherpa SenseVoice ASR | no (234 MB) | `voice_processing/agent_assets/models/asr/` | GitHub `k2-fsa/sherpa-onnx` releases | `fetch_assets sherpa` |
| sherpa KWS (zipformer) | no (~40 MB) | `voice_processing/agent_assets/models/kws/` | GitHub `k2-fsa/sherpa-onnx` releases | `fetch_assets sherpa` |
| **faster-whisper** (CTranslate2) vi/en/ja, GPU | no (~3 GB, large-v3) | HF cache (`models--Systran--faster-whisper-<size>`) | HF `Systran/faster-whisper-<size>` | `fetch_assets asr-fw` |
| **Whisper .pt** (nguồn build TensorRT) | no (~3.1 GB, large-v3) | `voice_processing/agent_assets/models/whisper/` | `openaipublic.azureedge.net` (sha256 pin) | `fetch_assets whisper-pt` |
| **ClearVoice** separation/TSE (MossFormer2_SS_16K) | no (~670 MB) | `voice_processing/checkpoints/` | HF `alibabasglab/MossFormer2_SS_16K` | `fetch_assets tse` |
| **pyannote** diarization/embedding | no | HF cache | HF `pyannote/*` (**gated**, cần `HF_TOKEN`) | `fetch_assets speaker` |
| Whisper multilingual ASR (vi/en/ja) | no (~360 MB) | `voice_processing/agent_assets/models/asr_whisper/` | HF `csukuangfj/sherpa-onnx-whisper-small` | `fetch_assets asr-whisper` |
| Vietnamese TTS (Piper vi_VN) | no (~60 MB) | `voice_processing/agent_assets/models/tts/` | GitHub `k2-fsa/sherpa-onnx` releases | `fetch_assets tts-vi` |
| Silero VAD `silero_vad.onnx` | **yes** (2.3 MB) | `voice_processing/agent_assets/models/` | `silero-vad` pip package | `fetch_assets silero` |
| Wake words `hey_*.onnx` | **yes** (~0.8 MB) | `voice_processing/agent_assets/models/` | custom-trained (openWakeWord) | tracked |
| Jetson wheels — torch 2.8.0, torchvision 0.23.0, torchaudio 2.8.0, onnxruntime_gpu 1.23.0 | no (~317 MB) | `voice_processing/wheels/` | Jetson index `pypi.jetson-ai-lab.io/jp6/cu126` (JetPack 6 / CUDA 12.6) | `fetch_assets wheels` |

## Notes

- The GGUF is a public stand-in (`unsloth/Qwen3.5-4B-GGUF`). If you need the
  team's private fine-tune instead, drop it into `llama-cpp/models/` and set
  `LLM_MODEL_FILE` in `.env` accordingly.
- `fetch_assets asr-fw` prefetches the **faster-whisper** (CTranslate2) model
  `WHISPER_MODEL` (default `large-v3`; also `medium`/`small`) into the
  HuggingFace cache (`models--Systran--faster-whisper-<size>`), so the unified
  `STT_BACKEND=whisper` works offline. It is the x86 GPU runtime; Jetson uses
  TensorRT (`whisper_trt`) with the same model name. To use a local directory
  instead, set `FASTER_WHISPER_MODEL` to an existing path.
- `fetch_assets whisper-pt` downloads the OpenAI **Whisper `.pt`** (sha256-pinned
  from the `openaipublic` CDN) used by the Jetson TensorRT runtime. On first run
  `whisper_trt` converts it to a TRT engine (10–20 min) and caches the engine in
  `/app/cache/whisper_trt`; later runs load the engine directly. `WHISPER_MODEL`
  selects the checkpoint (default `large-v3`).
- `fetch_assets asr-whisper` downloads the size in `STT_WHISPER_MODEL`
  (default `small`; also `tiny`/`base`/`medium`). Whisper is multilingual with
  automatic language detection but **slower** than SenseVoice — measured on CPU
  (int8, 2 threads, 7.15 s audio): SenseVoice 0.20 s (RTF 0.03) vs Whisper-small
  1.94 s (RTF 0.27). Use `base`/`tiny` for lower latency.
- `fetch_assets wheels` downloads each wheel by pinned URL + **sha256** from the
  NVIDIA Jetson index (`pypi.jetson-ai-lab.io/jp6/cu126`); no `pip` is required,
  and a mismatched build is detected and replaced. The pinned versions/sizes
  match the known-good r36.4 (JetPack 6.2) bundle. Override the host with
  `WHEELS_INDEX=<url>` if needed.
- `verify` (also run automatically after any fetch) checks every expected path
  and minimum size:

  ```bash
  scripts/fetch_assets.sh verify
  ```

- `voice_processing/kokoro_tts/.env` holds a personal OpenAI key for the
  standalone client — it is gitignored and never shipped. A `.env.example`
  template is provided.

## Benchmark datasets (auto-downloaded, no fetch needed)

Used by `scripts/asr_benchmark.py` (see [`BENCHMARK.md`](BENCHMARK.md)):

| Dataset | Source | Purpose |
|---------|--------|---------|
| `google/fleurs` (vi_vn, ja_jp, en_us, `test`) | HF | clean multilingual WER/CER + language detection |
| `corypaik/musan` (`noise`) | HF (mirror of OpenSLR MUSAN, CC BY 4.0) | additive noise at SNR levels |

They download to the HuggingFace cache on first run (`~/.cache/huggingface`).
