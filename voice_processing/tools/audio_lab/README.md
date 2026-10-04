# Audio Lab — preprocessing A/B

Web UI để **thu âm (hoặc tải WAV)** rồi so sánh **trước vs sau** chuỗi tiền xử lý
của `voice_processing`: WebRTC APM (**AEC · NS · HPF · transient · AGC2**, tuỳ chọn
**AGC1**) + **denoise** (spectral gating), kèm nghe A/B và visualize.

## Chạy

```bash
pip install -r voice_processing/tools/audio_lab/requirements.txt
python voice_processing/tools/audio_lab/app.py            # http://localhost:8020
# tùy chọn: --host 0.0.0.0 --port 8020
```

Mở `http://localhost:8020`:
1. **Thu âm** N giây (hoặc tải file WAV).
2. Chỉnh tuỳ chọn (NS level, AGC2 gain, AGC1 target-level, denoise…).
3. **Chạy so sánh** → xem waveform, spectrogram, mức RMS/peak (dBFS) cạnh nhau
   và nghe RAW vs PROCESSED.

## Định dạng file

UI nhận `audio/*`. Ngoài WAV/FLAC/OGG (libsndfile), các file nén `.m4a`/`.mp4`/`.mp3`
được giải mã qua **ffmpeg** (`pip install` không cần, chỉ cần ffmpeg trong PATH —
đã có sẵn trên máy dev). Sample rate và số kênh được giữ nguyên khi giải mã.

## Ghi chú

- **AEC thật cần far-end (âm loa đang phát)**. Ở lab offline không có playback
  reference nên AEC không có gì để khử; NS/HPF/transient/AGC/denoise vẫn áp dụng.
  Muốn thử AEC: chạy lab khi xe đang phát TTS (hoặc để dành kiểm tra trên thiết bị).
- File ghi âm lưu ở `tools/audio_lab/recordings/`.
- `Denoise` dùng `noisereduce` (spectral gating, chạy CPU). Nếu chưa cài thì toggle
  sẽ báo "không" ở góc phải header.
- Các tham số ánh xạ trực tiếp sang biến môi trường trong `aec.py`
  (`AEC_NS_LEVEL`, `AEC_AGC_MAX_GAIN_DB`, `AEC_AGC1_*`, `AEC_LIMITER_*`) — thử ở
  đây rồi chép giá trị ưng ý vào `.env`/`.env.x86`.
- **Limiter luôn bật mặc định** (`AEC_LIMITER_ENABLED=1`, trần `AEC_LIMITER_CEILING_DBFS=-1`).
  AGC1/AGC2 chỉ tăng gain, không có limiter, nên mic yếu sẽ đẩy peak lên 0 dBFS và
  méo. Tắt limiter để thấy rõ hiệu ứng clipping.
- **Target-level (AGC1) hiện do `aec.py` tự hiện thực**, không dùng
  `gain_control1` của WebRTC: field `target_level_dbfs` trong binding py-xiaozhi
  không có tác dụng (đặt −3 và −12 cho output giống hệt từng bit) và AGC1 không
  có limiter.
- Audio được nạp theo block giống runtime thật, nên slider phản ánh đúng như khi
  chạy mic thật.
