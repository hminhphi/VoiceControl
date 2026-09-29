#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for callers that explicitly request handsfree mode.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BLUETOOTH_AUDIO_PROFILE=hfu
exec "$PROJECT_DIR/setup_blue.sh" "$@"
