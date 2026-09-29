# Audio Setup — x86 Host (Windows + Docker Desktop)

Docker Desktop chạy Linux containers qua WSL2. Có **3 cách** để route audio vào container:

## Option A: ALSA trực tiếp qua `/dev/snd` (Đơn giản nhất)

Docker Desktop WSL2 expose `/dev/snd` từ kernel Linux. Không cần cấu hình thêm.

**Kiểm tra `/dev/snd` có không trong WSL2:**
```bash
# Trong PowerShell:
wsl -d docker-desktop -- ls /dev/snd
```
Nếu thấy `controlC0`, `pcmC0D0p`, … → ALSA available.

**Trong `docker-compose.x86.yml` (đã có):**
```yaml
devices:
  - /dev/snd:/dev/snd
group_add:
  - audio
```

**Trong `.env.x86`:**
```env
MIC_DEVICE_NAME=default   # hoặc tên thiết bị cụ thể
AUDIO_SAMPLE_RATE=48000
```

---

## Option B: PulseAudio qua TCP (Nếu có mic USB trên Windows)

Cài PulseAudio cho Windows và expose qua TCP:

1. Cài **PulseAudio for Windows** hoặc dùng `scoop install pulseaudio`
2. Sửa `C:\PulseAudio\etc\pulse\default.pa`:
   ```
   load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1;172.16.0.0/12 auth-anonymous=1
   ```
3. Trong `.env.x86`:
   ```env
   PULSE_SERVER=tcp:host.docker.internal:4713
   MIC_DEVICE_NAME=pulse
   ```
4. Trong `docker-compose.x86.yml` uncomment:
   ```yaml
   environment:
     - PULSE_SERVER=tcp:host.docker.internal:4713
   ```

---

## Option C: Chỉ test STT/TTS không cần mic thật (Nhanh nhất)

Dùng file WAV trong `input_test/` để test pipeline mà không cần audio hardware:

```bash
# Để sẵn WAV file vào:
cp your_audio.wav voice_processing/input_test/

# Chạy test:
docker compose -f docker-compose.x86.yml run --rm voice_processing \
    python3 test_voice_pipeline.py --wav /app/input_test/your_audio.wav
```

---

## Kiểm tra nhanh audio trong container

```bash
docker compose -f docker-compose.x86.yml run --rm \
    --device /dev/snd voice_processing \
    python3 -c "import sounddevice as sd; print(sd.query_devices())"
```
