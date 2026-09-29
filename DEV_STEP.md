Bước 0 — Kiểm tra nền tảng
df -h /                      # cần ~25-30GB trống cho unzip + images
docker info | grep -i runtime   # phải có nvidia runtime
docker images                # xác nhận sạch (kể cả l4t-jetpack đã bị prune)
Bước 1 — Copy & giải nén
# copy file từ máy dev lên (scp/rsync/USB)
unzip orchestrator-on-edge-jetson.zip -d ~/
cd ~/orchestrator-on-edge
ls -la .env.example run_all.sh docker-compose.yml   # phải thấy đủ
cp .env.example .env && nano .env                   # điền secret: GRAPHQL_*, SUDO_PWD, SPEAKER_MAC_ADDRESS...
Zip đã chứa sẵn: code arm64, models (sherpa/kokoro/silero/wake), cache/ (sentence-transformers), GGUF Qwen3.5-4B, wheels aarch64 và .env.example (AEC/16k). Zip KHÔNG chứa .env thật (secret) — phải copy từ .env.example.
Bước 2 — Sạch biến môi trường shell (bài học LOCAL_LLM_URL)
env | grep -E "LOCAL_LLM|OPENAI|AEC_|AUDIO_"   # nếu có thì unset trước khi build/run
unset LOCAL_LLM_URL LOCAL_LLM_MODEL
Lý do: host env đè .env của compose → container nhận sai giá trị.
Bước 3 — Pull base NVIDIA (nếu prune đã xóa)
docker pull nvcr.io/nvidia/l4t-jetpack:r36.4.0     # ~5.2GB tải, ~15.5GB trên disk
Bước 4 — Build base dùng chung (BẮT BUỘC trước, lỗi hay gặp nếu bỏ)
docker build -f Dockerfile.l4t-base -t orchestrator-on-edge/l4t-base:r36.4.0-torch2.8 .
docker images orchestrator-on-edge/l4t-base        # phải thấy tag r36.4.0-torch2.8
Bước 5 — Build 4 service (cần internet cho pip lúc này)
docker compose -p orch_v1 build voice_processing
docker compose -p orch_v1 build orchestrator car_control car_manual
Bỏ car_control_ui (Jetson dùng xe thật). WARN OPENAI_MODEL/INFOTAINMENT_MEDIA_DIR unset: bỏ qua.
Bước 6 — Dọn build-cache lấy lại vài GB
docker builder prune -f
docker system df
Bước 7 — Smoke test không audio trước
curl -s localhost:8000/health   # sau khi start — làm ở bước 8
Bước 8 — Chạy toàn bộ
chmod +x run_all.sh
./run_all.sh
run_all.sh tự: setup bluetooth/audio (setup_blue.sh), mở llama-server (native), rồi docker compose -p orch_v1 up --no-build.
Bước 9 — Verify
tmux attach -t orch_edge        # xem các window: llama / bluetooth / stack / audit
curl localhost:8000/health localhost:8001/health localhost:8002/health
Trong log voice_processing (window stack) phải thấy [AEC] Active ... agc=True ns=high.
Bước 10 — Cài systemd (tùy chọn, auto-start boot)
sudo cp orchestrator-on-edge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now orchestrator-on-edge
Lưu ý quan trọng
- Build cần internet (pip tải transformers/sentence-transformers/sherpa/kokoro…); wheel torch/onnxruntime thì lấy local trong repo. Sau build mới chạy offline được.
- Thứ tự bắt buộc: l4t-jetpack → l4t-base → services. Bỏ bước 4 sẽ lỗi pull access denied.
- Disk lúc build sẽ peak (context + layer tạm) rồi giảm sau prune; theo dõi df -h nếu sát 30GB.
- Nếu log báo [AEC] passthrough thì .so aarch64 không load — tạm đặt AEC_ENABLED=0 trong .env.
- Mic nếu Pulse không mở được 16k: revert AUDIO_SAMPLE_RATE=48000 AUDIO_CHUNK=2048 trong .env. 