#!/usr/bin/env bash
set -euo pipefail

# Kiểm tra pigz đã cài chưa
if ! command -v pigz &> /dev/null; then
    echo "Error: pigz not found. Install with: sudo apt-get install pigz"
    exit 1
fi

DIR="/media/nvidia/Kingston/backup_0410"
mkdir -p "$DIR"

# Lấy danh sách images (loại bỏ <none>)
mapfile -t IMAGES < <(
  docker image ls --format '{{.Repository}}:{{.Tag}}' \
  | grep -v '^<none>:' \
  | sort -u
)

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "Error: No Docker images found"
    exit 1
fi

printf '%s\n' "${IMAGES[@]}" > "$DIR/image-list.txt"
echo "Backing up ${#IMAGES[@]} images..."

# Save + nén với pigz (parallel gzip, mặc định dùng tất cả cores)
docker image save "${IMAGES[@]}" \
  | pigz -1 \
  > "$DIR/all-images-arm64.tar.gz"

# Kiểm tra integrity
pigz -t "$DIR/all-images-arm64.tar.gz" &&
sha256sum "$DIR/all-images-arm64.tar.gz" \
  > "$DIR/all-images-arm64.tar.gz.sha256"

echo ""
ls -lh "$DIR"
echo ""
echo "Backup complete!"
echo "SHA256: $(cat $DIR/all-images-arm64.tar.gz.sha256)"
