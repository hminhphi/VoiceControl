vendor/apt-debs — bộ .deb cho build OFFLINE trên Jetson
=========================================================
Các file .deb trong thư mục này được pack_jetson.py đóng vào bundle nhưng
KHÔNG được commit vào git (xem .gitignore).

Nguồn: tải từ đúng image nvcr.io/nvidia/l4t-jetpack:r36.4.0 bằng
  apt-get install --download-only
cho các gói mà Dockerfile.l4t-base / voice_processing cần mà image chưa có
(pkg-config, portaudio19-dev, libsndfile1, libasound2-plugins, alsa-utils,
pulseaudio-utils, ffmpeg, git, wget, curl, python3-pip, libopenblas0, ...).

Dockerfile.l4t-base kiểm tra thư mục này: nếu có *.deb -> `dpkg -i` (offline),
nếu không -> fallback `apt-get update && apt-get install` (cần mạng 1 lần).
