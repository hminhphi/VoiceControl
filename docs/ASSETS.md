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
| Silero VAD `silero_vad.onnx` | **yes** (2.3 MB) | `voice_processing/agent_assets/models/` | `silero-vad` pip package | `fetch_assets silero` |
| Wake words `hey_*.onnx` | **yes** (~0.8 MB) | `voice_processing/agent_assets/models/` | custom-trained (openWakeWord) | tracked |
| Jetson wheels — torch 2.8.0, torchvision 0.23.0, torchaudio 2.8.0, onnxruntime_gpu 1.23.0 | no (~317 MB) | `voice_processing/wheels/` | Jetson index `pypi.jetson-ai-lab.io/jp6/cu126` (JetPack 6 / CUDA 12.6) | `fetch_assets wheels` |

## Notes

- The GGUF is a public stand-in (`unsloth/Qwen3.5-4B-GGUF`). If you need the
  team's private fine-tune instead, drop it into `llama-cpp/models/` and set
  `LLM_MODEL_FILE` in `.env` accordingly.
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
