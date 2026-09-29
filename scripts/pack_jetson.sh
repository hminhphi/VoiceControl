#!/usr/bin/env bash
# Build a ready-to-extract Jetson (arm64) deployment zip.
# Usage: scripts/pack_jetson.sh [--out dist] [--name orchestrator-on-edge]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

exec "$PY" "$SCRIPT_DIR/pack_jetson.py" "$@"
