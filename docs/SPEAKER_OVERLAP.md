# Speaker Overlap — nghiên cứu & kế hoạch tích hợp

Nghiên cứu cách xử lý **nhiều người nói chồng tiếng** (overlapped speech /
"cocktail party") cho `orchestrator-on-edge`, dựa trên các repo đã khảo sát ở
`github-prior-art`. Kèm PoC: `voice_processing/tools/speaker_overlap.py`.

## 1. Vấn đề

Pipeline hiện tại (1 micro) giả định **một người nói tại một thời điểm**. Khi có
2+ người nói chồng tiếng:

- ASR (Whisper) nhận **hỗn hợp** 2 giọng → transcript sai.
- VAD chỉ biết "có tiếng nói", **không biết ai** đang nói.
- Wake word (openWakeWord) có thể kích nhầm khi người khác nói câu giống.

Mục tiêu: **nhận biết có chồng tiếng**, và **ưu tiên người đang nói với xe**
(người đã gọi wake word), thay vì trộn tất cả.

## 2. Kiến trúc lý tưởng

```
mic ─▶ [AEC/NS/AGC] ─▶ VAD ─▶ OSD (overlap?) ─▶ Diarization ─▶ TSE ─▶ ASR ─▶ LLM
                                  │                              ▲
                                  └─ nếu overlap ────────────────┘
                                     (tách giọng target)
        wake word ─▶ enrollment (voiceprint) ─┘
```

- **OSD (Overlapped Speech Detection):** phát hiện đoạn có ≥2 người nói.
- **Diarization (ai nói khi nào):** gán nhãn người nói theo thời gian.
- **TSE (Target Speaker Extraction):** trích **giọng mục tiêu** khỏi hỗn hợp, dùng
  wake word làm **enrollment** (reference) → đưa audio sạch cho ASR.
- **"Người gần/to nhất":** cần **mảng mic + DOA** (Direction of Arrival) —
  1 mic không làm được; thay thế bằng **TSE + wake-word gating**.

## 3. So sánh các dự án (nguồn: github-prior-art)

| Dự án | License | Có gì | Model/Offline | Ghi chú tích hợp |
|-------|---------|-------|---------------|------------------|
| **modelscope/3D-Speaker** ★3.1k | Apache-2.0 | VAD + **overlap detection** + diarization + speaker embedding (CAM++/ERes2Net) + **ONNX Runtime** | ModelScope; ONNX chạy CPU/GPU | ⭐ Tốt nhất cho "ai nói khi nào + overlap", license thoáng, có runtime ONNX |
| **pyannote/pyannote-audio** ★10.6k | MIT (code) / model gated | SOTA diarization + VAD + segmentation (overlap) | HF `pyannote/speaker-diarization-community-1` (**cần token + accept điều khoản**) | Chất lượng cao, nhưng model **gated**, cần ffmpeg/torch |
| **modelscope/ClearerVoice-Studio** ★4.5k | Apache-2.0 | **Speech separation** (MossFormer) + **Target Speaker Extraction** (audio-only theo reference speech) + denoise (FRCRN/DPDFNet) | `pip install clearvoice`; ModelScope/HF | ⭐ Dùng cho **TSE** (tách giọng người gọi xe) và tách 2 người nói chồng |
| **speechbrain/speechbrain** ★11.8k | Apache-2.0 | SepFormer/DPRNN separation, TSE recipes | HF; torch | Nền tảng thuật toán; nặng hơn ClearerVoice khi chỉ cần inference |
| **modelscope/FunASR** ★20.5k | MIT | ASR + VAD + punctuation + diarization + streaming | ModelScope | All-in-one nếu muốn gộp ASR+diarization |
| **QuentinFuxa/WhisperLiveKit** ★11.1k | Apache-2.0 | Streaming ASR + diarization real-time | whisper + diarization | Tham khảo kiến trúc streaming |

## 4. Khuyến nghị cho repo này

Với **1 micro** (hiện tại), hướng thực tế và hiệu quả nhất:

1. **Wake-word as enrollment:** người gọi "Hey Dora" chính là **target**.
2. **OSD + diarization** (3D-Speaker, Apache-2.0, ONNX) để biết đoạn nào chồng
   tiếng và ai đang nói — dùng để **quyết định**: nếu overlap thì ưu tiên target,
   nếu target im lặng thì "xin nhắc lại".
3. **TSE** (ClearerVoice-Studio, audio-only conditioned on reference speech):
   lấy **mẫu giọng wake-word** làm reference → **tách** giọng target khỏi hỗn hợp
   → đưa cho ASR. Đây là thay thế khả thi cho beamforming khi chỉ có 1 mic.
4. (Tùy chọn) `pyannote` nếu chấp nhận model gated — chất lượng diarization cao
   nhất, hợp để benchmark/đối chiếu.

**Vì sao không "chọn người to nhất" đơn thuần:** năng lượng lớn nhất không = người
đang giao tiếp với xe (có thể là người bên cạnh nói to). Wake-word enrollment +
TSE định danh đúng mục tiêu hơn.

## 5. Điểm tích hợp (khi triển khai)

Chèn giữa VAD và STT, gated bằng env (mặc định tắt để không đổi hành vi):

```
AUDIO → VAD (đã có) → [SPEAKER_STAGE] → STT
                        ├─ SPEAKER_OSD=1        (3D-Speaker OSD)
                        ├─ SPEAKER_TSE=1        (ClearerVoice TSE, reference=wake-word clip)
                        └─ SPEAKER_HF_TOKEN=... (nếu dùng 3D-Speaker include_overlap)
```

- Lưu **~1s audio sau wake word** làm reference cho TSE (enrollment).
- Nếu OSD báo overlap **và** target im lặng → gửi TTS "Tôi nghe nhiều người cùng
  lúc, bạn nhắc lại giúp tôi".
- Model + ONNX tải bằng `scripts/fetch_assets` (thêm subcommand `speaker`) —
  đặt dưới `voice_processing/agent_assets/models/speaker/` (gitignored).
- **Độ trễ:** OSD/diarization offline trên đoạn utterance (~0.2–0.5s cho model
  nhỏ, ONNX CPU); TSE thêm ~0.1–0.3s. Chấp nhận được nếu chỉ chạy khi OSD bật.

## 6. PoC

`voice_processing/tools/speaker_overlap.py`:

```bash
# Tự dò backend (pyannote / 3D-Speaker); báo hướng dẫn nếu chưa cài
python voice_processing/tools/speaker_overlap.py --wav mixed.wav
# pyannote (cần HF token + accept điều khoản model)
python voice_processing/tools/speaker_overlap.py --wav mixed.wav --backend pyannote --hf-token hf_xxx
```

PoC xuất **timeline người nói**, **đoạn overlap** (≥2 speaker cùng lúc), thống kê
% overlap, và ghi file `.rttm`. Nó cũng hướng dẫn dùng **ClearerVoice** để tách
giọng target khi cần.

## 7. License & rủi ro

- **3D-Speaker** Apache-2.0, **ClearerVoice-Studio** Apache-2.0, **FunASR** MIT →
  dùng thương mại được (giữ attribution).
- **pyannote**: code MIT nhưng **model trên HF yêu cầu token + điều khoản** (không
  tự do hoàn toàn) → cân nhắc.
- Model diarization/TSE **nặng** (torch/ONNX + GPU), tăng **độ phức tạp & latency**
  → nên để **opt-in**, không bật mặc định.
- Chưa có **mic array** → không thể "chọn người gần/hướng"; giải pháp thay thế là
  **wake-word enrollment + TSE**.

## 8. Bước tiếp theo đề xuất

1. Thêm subcommand `fetch_assets speaker` tải model 3D-Speaker ONNX (+ ClearerVoice).
2. Thêm stage `SPEAKER_OSD`/`SPEAKER_TSE` trong `voice_processing` (opt-in).
3. Đo trên audio thật (2 người nói chồng) bằng PoC → tinh chỉnh ngưỡng.
4. (Tùy chọn) benchmark đối chiếu với pyannote.

## 9. Đã triển khai (opt-in, GPU)

- **`voice_processing/speaker.py`** — `SpeakerAnalyzer` (pyannote.audio **trên torch CUDA**,
  chạy GPU cả x86 `cu128` lẫn Jetson `torch-2.8 aarch64`): diarization →
  **overlap intervals** → dominant speaker; kèm embedding để **chọn người nói mục tiêu**.
  Tự tắt an toàn nếu thiếu pyannote/token.
- **`voice_processing/main.py`** — `process_stt` gom audio từng turn, chạy analyzer,
  log `[SPEAKER] overlap=...`; nếu `SPEAKER_GATE_OVERLAP=1` và overlap ≥ ngưỡng →
  phát câu "nhiều người nói, nhắc lại" (theo ngôn ngữ) thay vì đẩy audio lẫn vào LLM.
- **Docker**: `voice_processing/Dockerfile` (arm64) + `Dockerfile.x86` cài
  `pyannote.audio` (dùng torch CUDA sẵn có). `ffmpeg` đã có trong base.
- **PoC**: `tools/speaker_overlap.py` (pyannote/3D-Speaker + fallback VAD, xuất RTTM).

**Bật (mặc định tắt):**

```dotenv
SPEAKER_ENABLED=1
SPEAKER_DEVICE=cuda
SPEAKER_PIPELINE=pyannote/speaker-diarization-community-1
SPEAKER_EMBEDDING=pyannote/embedding
SPEAKER_HF_TOKEN=hf_xxx           # model pyannote là gated: accept điều khoản + token
SPEAKER_GATE_OVERLAP=1            # overlap >= ngưỡng -> xin nhắc lại
SPEAKER_OVERLAP_MIN=0.15
```

**TSE thật (tách giọng target khỏi hỗn hợp)** vẫn là bước mở rộng — dùng
`ClearerVoice-Studio` (Apache-2.0, torch CUDA) với reference = mẫu giọng wake-word;
chưa tích hợp (cần thêm model + đo trên thiết bị).
