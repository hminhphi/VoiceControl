#!/usr/bin/env bash
# Wrapper: download orchestrator-on-edge assets (see docs/ASSETS.md).
# Usage: scripts/fetch_assets.sh --all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

exec "$PY" "$SCRIPT_DIR/fetch_assets.py" "$@"
