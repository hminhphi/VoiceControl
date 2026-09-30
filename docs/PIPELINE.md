# Full Pipeline — chốt cấu hình & sơ đồ

Tài liệu này **chốt** trạng thái pipeline hiện tại và vẽ **full flow** bằng
[PlantUML](https://plantuml.com). Nguồn sơ đồ: [`pipeline.puml`](pipeline.puml).

## 1. Chốt cấu hình (defaults)

| Thành phần | x86 (dev GPU) | Jetson (arm64) |
|---|---|---|
| **STT** | `faster_whisper` (CTranslate2, `large-v3`, CUDA float16) | `sherpa_onnx`/SenseVoice (mặc định repo) — đổi sang `faster_whisper` khi cần |
| **ASR fallback** | `sherpa_whisper` (CPU), `sherpa_onnx` | `whisper_trt` (TensorRT) / `sherpa_onnx` |
| **TTS** | Kokoro (en `af_heart`, ja `jf_alpha` + `misaki[ja]`), **Piper vi_VN** | Kokoro (en), Piper vi_VN (nếu fetch) |
| **Ngôn ngữ** | `STT_LANGUAGE=auto`; LLM nhận **language hint** từ STT | như trên |
| **Preprocessing** | WebRTC APM: AEC + NS (moderate) + NS linear-AEC + HPF + transient + AGC2 (`max_noise=-18`) | như trên |
| **VAD** | Silero dual-threshold, `VAD_WINDOW_SIZE=3`, `TURN_END_SILENCE=1.0` | như trên |
| **Speaker stage** | opt-in: `SPEAKER_ENABLED`, `SPEAKER_TSE_ENABLED` (pyannote + ClearVoice, CUDA) | như trên (cần fetch model) |
| **Provider GPU** | `ONNX_PROVIDER`/`SHERPA_PROVIDER`/`VAD_PROVIDER` = `cpu` mặc định, đặt `cuda` khi có GPU build | như trên |
| **LLM** | llama-server (Qwen3.5-4B GGUF), tool-calling | như trên |

> GPU đã sẵn cho **TTS (Kokoro ORT CUDA)**, **ASR (faster-whisper CUDA / Whisper TensorRT)**, **diarization/TSE (torch CUDA)**. VAD/wake word để CPU (model nhỏ).

## 2. Các giai đoạn (stage) & biến bật/tắt

| # | Stage | Công cụ | Env chính |
|---|-------|---------|-----------|
| 1 | Tiền xử lý | WebRTC APM (`aec.py`) | `AEC_ENABLED`, `AEC_NS_LEVEL`, `AEC_NS_LINEAR`, `AEC_AGC_*`, `AEC_AGC1_*` |
| 2 | VAD | Silero (`vad.py`) | `VAD_THRESHOLD_HIGH/LOW`, `VAD_DETECT_THRESHOLD`, `VAD_WINDOW_SIZE` |
| 3 | Wake word | openWakeWord / sherpa KWS | `WAKE_WORD_BACKEND`, `WAKE_WORD_THRESHOLD` |
| 4 | Cắt lượt | `main.py` | `SEGMENT_SILENCE`, `TURN_END_SILENCE`, `MIN_SEGMENT_SEC`, `STT_MIN_RMS` |
| 5 | STT | `faster_whisper` / `sherpa_whisper` / SenseVoice | `STT_BACKEND`, `STT_LANGUAGE`, `FASTER_WHISPER_*`, `STT_WHISPER_*` |
| 6 | Lọc hallucination | `stt.py` | `STT_DROP_SHORT_HALLUCINATIONS`, `STT_HALLUCINATION_BLOCKLIST`, `FW_*` |
| 7 | Speaker (opt-in) | pyannote (`speaker.py`) | `SPEAKER_ENABLED`, `SPEAKER_GATE_OVERLAP`, `SPEAKER_OVERLAP_MIN` |
| 8 | TSE (opt-in) | ClearVoice (`tse.py`) | `SPEAKER_TSE_ENABLED`, `TSE_MODEL`, `TSE_REF_SEC` |
| 9 | LLM | orchestrator (`route_orchestrator.py`) | `LOCAL_LLM_URL`, `language` hint, `_language_directive` |
| 10 | TTS | Kokoro + Piper (`tts.py`) | `TTS_LANGUAGE`, `TTS_VOICE_*`, `PIPER_VI_DIR` |

## 3. Full flow (PlantUML)

### 3.1 Runtime — voice pipeline

```plantuml
@startuml Runtime - voice pipeline (GPU-first)
|#E3F2FD|Audio in|
start
:Capture mic @ AUDIO_SAMPLE_RATE (sounddevice);
:AEC · NS · HPF · transient · AGC2 (WebRTC APM);
note right: AEC khử echo (cần far-end = loa); NS/AGC nâng giọng cho model
|#E8F5E9|VAD + Wake|
:Silero VAD (dual-threshold, VAD_WINDOW_SIZE);
if (wake word HOẶC follow-up?) then (yes)
  :capture command (pre-roll SPEECH_PREROLL_SEC);
|#FFF3E0|STT|
  while (im lặng < TURN_END_SILENCE) is (đang nói)
    if (im lặng ≥ SEGMENT_SILENCE) then (yes)
      if (RMS ≥ STT_MIN_RMS) then (yes)
        :STT faster-whisper large-v3 (CUDA) · vi/en/ja · auto;
        :lọc hallucination;
      else (no)
        :bỏ đoạn năng lượng thấp;
      endif
    endif
  endwhile
|#F3E5F5|Speaker (opt-in)|
  if (SPEAKER_ENABLED?) then (yes)
    :diarization + overlap (pyannote CUDA);
  endif
  if (overlap ≥ SPEAKER_OVERLAP_MIN?) then (yes)
    if (SPEAKER_TSE_ENABLED?) then (yes)
      :TSE separation (ClearVoice CUDA) → chọn target;
      :STT lại trên stream sạch;
    else (no)
      if (SPEAKER_GATE_OVERLAP?) then (yes)
        :TTS xin nhắc lại; stop
      endif
    endif
  endif
|#E1F5FE|Orchestrator + LLM|
  :POST /v1/orchestrator/message (text + language);
  :LLM tool-calling (llama-server) + language directive;
  if (tool call?) then (yes)
    :car_control / car_manual (A2A);
  endif
  :stream tokens (WebSocket);
|#FFF8E1|TTS + playback|
  while (token stream) is (còn câu)
    :gom câu → TTS (Kokoro en/ja · Piper vi);
  endwhile
  :playback (barge-in, follow-up 12s);
else (no)
  :chờ wake word;
endif
stop
@enduml
```

### 3.2 Deployment — services & ports

```plantuml
@startuml Deployment - services & ports
actor "Người dùng" as U
rectangle "voice_processing\n(host audio · GPU CUDA)" as VP
rectangle "Orchestrator :8000" as ORCH
database "llama-server :8080\nQwen3.5-4B GGUF" as LLM
rectangle "car_control :8001" as CC
rectangle "car_manual :8002" as CM
rectangle "navigation :8003" as NAV
rectangle "infotainment :8004" as INFO
rectangle "cloud :8005" as CL
rectangle "car_control_ui :8010 (dev)" as UI
rectangle "frontend :3000 (opt)" as FE
U --> VP : mic / loa
VP --> ORCH : HTTP + WebSocket (text + language)
ORCH --> LLM : OpenAI-compatible
ORCH --> CC
ORCH --> CM
ORCH --> NAV
ORCH --> INFO
ORCH --> CL
CC --> UI
FE --> ORCH
@enduml
```

### 3.3 Build & deploy — asset → bundle → run

```plantuml
@startuml Build and deploy
start
:scripts/fetch_assets (llm · embed · kokoro · asr-fw · tts-vi · tse · wheels);
:scripts/pack_jetson (arm64, loại x86-only);
:dist/orchestrator-on-edge-jetson-<date>.zip;
:copy lên Jetson (scp / USB);
:unzip · cp .env.example .env;
:build l4t-jetpack → l4t-base → services;
:./run_all.sh (audio · llama-server · compose);
:tmux orch_edge;
stop
@enduml
```

## 4. Render PlantUML

- **GitLab**: render trực tiếp khối ` ```plantuml ` trong markdown; hoặc commit `pipeline.puml` (GitLab hiển thị).
- **GitHub**: không render sẵn — dùng [PlantUML proxy](https://github.com/plantuml/plantuml-server) / extension VSCode `jebbs.plantuml`, hoặc PlantUML CLI:
  ```bash
  plantuml -tpng docs/pipeline.puml
  ```
- Nguồn gộp 3 sơ đồ: [`pipeline.puml`](pipeline.puml).

## 5. Trạng thái & việc còn lại

- **Đã chốt:** STT đa ngữ vi/en/ja (GPU), chống ồn/ hallucination, TTS theo ngôn ngữ, LLM trả lời đúng tiếng (language hint), speaker diarization + TSE (opt-in), tools (Audio Lab, VAD tuner, benchmark).
- **Kết quả benchmark:** xem [`BENCHMARK.md`](BENCHMARK.md).
- **Còn lại:** (1) bật pyannote embedding để chọn target chính xác; (2) tách VRAM khi chạy TSE chung large-v3; (3) benchmark 2 giọng cùng ngôn ngữ.
