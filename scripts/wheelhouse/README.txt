scripts/wheelhouse — lock cho wheelhouse offline (voice_processing/wheels)
============================================================================
Các file lock này là "công thức" tạo wheelhouse aarch64 (python 3.10) dùng cho
build OFFLINE trên Jetson (`pip install --no-index --find-links /tmp/wheels`).
Chúng ĐƯỢC COMMIT vào git; wheelhouse thật (nhiều GB) thì KHÔNG (xem .gitignore).

  lock_base_full_clean.txt   deps của l4t-base (numpy/scipy/sherpa/torch/onnxruntime-gpu/uv...)
  lock_voice_clean.txt       deps của voice_processing (transformers 5.x, pyannote,
                             librosa==0.10.2.post1 (ghim theo clearvoice), misaki, Cython...)
  lock_orch_clean.txt        deps của orchestrator/car_control/car_manual
                             (transformers==4.46.3, sentence-transformers, a2a-sdk...)
  lock_jtalk_clean.txt       deps của pyopenjtalk-plus (sudachipy, sudachidict-core...)
  req_sdists.txt             7 gói CHỈ có sdist (PyAudio, pesq, mojimoji...) — build
                             trên Jetson (gcc) hoặc build wheel sẵn trên x86 (xem dưới)
  constraints.txt            ghim chung khi resolve (torch==2.8.0 local wheel,
                             numpy==1.26.4, scipy==1.15.3, onnxruntime-gpu==1.23.0)

TẠO LẠI wheelhouse từ đầu (máy x86, có mạng):
  # 1. Tải toàn bộ wheel aarch64 (không --no-deps để lấy đủ cây):
  pip download -d voice_processing/wheels \
    --platform manylinux2014_aarch64 --platform manylinux_2_17_aarch64 \
    --platform manylinux_2_24_aarch64 --platform manylinux_2_28_aarch64 \
    --platform manylinux_2_31_aarch64 --platform manylinux_2_35_aarch64 \
    --platform manylinux_2_36_aarch64 --platform linux_aarch64 \
    --implementation cp --python-version 3.10 --abi cp310 --no-deps \
    -r scripts/wheelhouse/lock_base_full_clean.txt \
    -r scripts/wheelhouse/lock_voice_clean.txt \
    -r scripts/wheelhouse/lock_orch_clean.txt \
    -r scripts/wheelhouse/lock_jtalk_clean.txt \
    -r scripts/wheelhouse/req_sdists.txt

  # 2. 4 gói pure-python cần wheel build sẵn (setuptools 2 kiểu — mâu thuẫn
  #    về bản build nên phải build riêng trên x86; xem Dockerfile voice §5/§9):
  pip install "setuptools==69.5.1" wheel && \
    pip wheel --no-deps --no-build-isolation -w voice_processing/wheels \
      voice_processing/wheels/openai-whisper-20240927.tar.gz \
      voice_processing/wheels/pystoi-*.tar.gz voice_processing/wheels/unidic-lite-*.tar.gz
  pip install "setuptools>=77,<81" && \
    pip wheel --no-deps --no-build-isolation -w voice_processing/wheels \
      voice_processing/wheels/clearvoice-0.1.2.tar.gz

  # 3. Kiểm tra đủ/không (thiếu là build offline sẽ fail):
  python scripts/verify_wheelhouse.py

LƯU Ý quan trọng (bài học từ lần build thực tế):
  - KHÔNG để wheel torch/torchaudio PyPI bản CPU (manylinux_2_28) lẫn vào —
    pip ưu tiên tag manylinux hơn linux_aarch64 → torchvision NVIDIA lệch ABI
    (lỗi `operator torchvision::nms does not exist`). Chỉ giữ wheel local
    `linux_aarch64` từ pypi.jetson-ai-lab.io.
  - pip download theo lock KHÔNG dùng constraints có thể tải nhầm numpy 2.x —
    kiểm lại bằng verify + xoá bản trùng (mỗi (name,version) chỉ 1 file).
