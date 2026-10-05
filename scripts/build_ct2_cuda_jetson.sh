#!/usr/bin/env bash
# Build CTranslate2 4.8.2 WITH CUDA for Jetson (aarch64) and emit the wheel +
# libctranslate2.so that voice_processing/Dockerfile installs.
#
# The PyPI aarch64 wheel is CPU-only, so this must be built from source.
# Run inside the voice_processing image so the ABI matches the runtime
# (Python 3.10 + CUDA 12.6 + cuDNN 9.3); on x86 this also works through
# QEMU emulation. Needs network only if the source/wheels are not mounted.
#
#   mkdir -p /tmp/ct2out
#   docker run --rm -v <src>:/src:ro -v <wheels>:/wheels:ro -v /tmp/ct2out:/out \
#     --entrypoint bash orchestrator-on-edge/voice_processing:latest /ct2build.sh
#
# Artifacts land in /out; copy them into voice_processing/wheels/.
# Build CTranslate2 4.8.2 voi CUDA cho aarch64, chay TRONG container ARM
# (QEMU emulation) tren may x86. Source + wheels mount tu bundle offline.
set -euo pipefail

CT2_VER="4.8.2"
CUDA_ARCH="8.7"
WORK=/tmp/ct2build
SRC=/src/CTranslate2
WHEELS=/wheels
OUT=/out
JOBS="${JOBS:-8}"
MAKELOG="$OUT/make.log"

test -d "$SRC"    || { echo "LOI: khong thay /src/CTranslate2"; exit 1; }
test -d "$WHEELS" || { echo "LOI: khong thay /wheels"; exit 1; }
mkdir -p "$OUT"
: > "$MAKELOG"

echo "[$(date +%H:%M:%S)] === [1/7] Cai build tools (offline) ==="
pip install --quiet --no-index --find-links "$WHEELS" \
    cmake ninja wheel setuptools pybind11
cmake --version | head -1
ninja --version

echo "[$(date +%H:%M:%S)] === [2/7] Copy source sang $WORK ==="
rm -rf "$WORK"; mkdir -p "$WORK"
cp -a "$SRC" "$WORK/CTranslate2"
cd "$WORK/CTranslate2"

echo "[$(date +%H:%M:%S)] === [3/7] CMake configure ==="
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_MKL=OFF \
    -DWITH_DNNL=OFF \
    -DWITH_OPENBLAS=OFF \
    -DWITH_RUY=ON \
    -DOPENMP_RUNTIME=COMP \
    -DCUDA_ARCH_LIST="${CUDA_ARCH}" \
    -DBUILD_CLI=OFF \
    -DBUILD_TESTS=OFF 2>&1 | tail -18
grep -iE "^WITH_CUDA|^WITH_CUDNN|^CUDA_ARCH_LIST" CMakeCache.txt || true

echo "[$(date +%H:%M:%S)] === [4/7] make -j${JOBS} (log: $MAKELOG) ==="
make -j"$JOBS" > "$MAKELOG" 2>&1
tail -5 "$MAKELOG"
make install > /dev/null
ldconfig

echo "[$(date +%H:%M:%S)] --- ldd check ---"
ldd /usr/local/lib/libctranslate2.so.${CT2_VER} | grep -iE "cublas|cudnn|cudart" \
    || { echo "LOI: libctranslate2 KHONG link CUDA"; exit 1; }

echo "[$(date +%H:%M:%S)] === [5/7] Python wheel ==="
cd "$WORK/CTranslate2/python"
pip install --quiet --no-index --find-links "$WHEELS" -r install_requirements.txt
python setup.py bdist_wheel 2>&1 | tail -6
WHL="$(ls dist/*.whl | head -1)"

echo "[$(date +%H:%M:%S)] === [6/7] Thu thap vao $OUT ==="
cp "$WHL" "$OUT/"
cp -P /usr/local/lib/libctranslate2.so* "$OUT/"
ls -la "$OUT"

echo "[$(date +%H:%M:%S)] === [7/7] Cai wheel + kiem tra ==="
pip install --quiet --force-reinstall --no-deps "$WHL"
python - <<'PY'
import ctranslate2
print("ctranslate2:", ctranslate2.__version__)
print("cuda_devices:", ctranslate2.get_cuda_device_count(), "(0 la binh thuong: container nay khong co GPU)")
PY

echo "[$(date +%H:%M:%S)] === BUILD_DONE ==="
