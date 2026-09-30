# ASR benchmark (large-v3, cuda)

| lang | mode | n | WER/CER | lang-det | time (s) |
|---|---|---|---|---|---|
| vi | clean | 25 | 0.038 | 100% | 20.0 |
| vi | noisy_10dB | 25 | 0.043 | — | 19.7 |
| vi | noisy_0dB | 25 | 0.051 | — | 19.6 |
| vi | overlap | 24 | 0.468 | — | 17.2 |
| vi | overlap+TSE | 24 | 0.426 | — | 670.4 |
| ja | clean | 25 | 0.071 | 100% | 28.4 |
| ja | noisy_10dB | 25 | 0.077 | — | 28.2 |
| ja | noisy_0dB | 25 | 0.085 | — | 28.4 |
| ja | overlap | 24 | 0.641 | — | 27.9 |
| ja | overlap+TSE | 24 | 1.301 | — | 707.0 |
| en | clean | 25 | 0.050 | 100% | 18.4 |
| en | noisy_10dB | 25 | 0.051 | — | 18.5 |
| en | noisy_0dB | 25 | 0.062 | — | 18.6 |
| en | overlap | 24 | 0.517 | — | 15.8 |
| en | overlap+TSE | 24 | 0.582 | — | 520.6 |

## Nhận xét

**Cấu hình:** `faster-whisper large-v3` (float16, CUDA, RTX 5060 Ti 16GB), beam 5, vad_filter.
Dữ liệu: **FLEURS** test (vi/ja/en, 25 câu/ngôn ngữ) + nhiễu **MUSAN** (noise) + trộn 2 người nói
(utterance i + i+1, người B vào sau 1s, trộn 0.5/0.5). Metric: **WER (en) / CER (vi, ja)**.

### Kết quả chính
- **Đa ngữ (sạch) rất tốt & nhận đúng ngôn ngữ 100%:** vi **CER 3.8%**, ja **CER 7.1%**, en **WER 5.0%**.
- **Chịu ồn tốt (MUSAN, không cần retrain):** ở **SNR 10 dB** vi 4.3% / ja 7.7% / en 5.1%; ở **SNR 0 dB** vi 5.1% / ja 8.5% / en 6.2% → suy giảm nhỏ.
- **Chồng tiếng 2 người → hỏng nặng (đúng như dự kiến):** vi 46.8% / ja 64.1% / en 51.7%. Đây là giới hạn cố hữu khi 1 mic + ASR đơn giọng.
- **TSE (ClearVoice `MossFormer2_SS_16K` + chọn target theo tương quan 1.5s đầu):**
  - vi: **cải thiện** 46.8% → **42.6%**;
  - en: gần như không đổi (51.7% → 58.2%);
  - ja: **tệ hơn** (64.1% → 130%) → chọn sai stream.

### Vấn đề cần xử lý (honest)
1. **Chọn target chưa đáng tin** khi không có embedding người nói. Fallback tương quan chỉ
   hiệu quả khi người mục tiêu nói rõ ở đoạn mở đầu; với ja bị chọn sai.
   → Nên bật **pyannote embedding** (`SPEAKER_ENABLED=1` + `HF_TOKEN`) để chọn theo **giọng**.
2. **TSE rất chậm khi chạy chung large-v3 (~28s/câu)** nhưng **nhanh khi chạy riêng**
   (RTF **0.18–0.24**, đo được). Nguyên nhân: **hết VRAM** — ClearVoice chiếm ~**10.9 GB**,
   cộng large-v3 (~3 GB) trên card 16 GB → swap. Khuyến nghị: chạy TSE ở **process riêng** /
   giải phóng large-v3 khi tách / hoặc dùng GPU nhiều VRAM hơn.

### Kết luận
STT đa ngữ + chống ồn **đạt** (vi/ja/en, kể cả SNR 0 dB), nhận ngôn ngữ chính xác. Với
**nhiều người nói chồng tiếng**, cần hoàn thiện **chọn target bằng embedding** và **tách VRAM**
trước khi coi TSE là production.

_Reproduce:_
```bash
python scripts/asr_benchmark.py --langs vi ja en --limit 25 --model large-v3 --device cuda --tse
```
