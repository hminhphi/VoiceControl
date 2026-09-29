#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_PROJECT="${COMPOSE_PROJECT:-orch_v1}"
TMUX_SESSION="${TMUX_SESSION:-orch_edge}"

_read_env_var() {
  local env_file="$1"
  local key="$2"
  [[ -f "$env_file" ]] || return 1
  local line
  line="$(
    sed -nE "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)[[:space:]]*$/\\1/p" "$env_file" | tail -n 1
  )"
  [[ -n "${line:-}" ]] || return 1
  line="${line%\"}"
  line="${line#\"}"
  line="${line%\'}"
  line="${line#\'}"
  printf '%s' "$line"
}

_enabled() {
  local value="${1:-}"
  value="${value,,}"
  [[ -n "$value" && "$value" != "0" && "$value" != "false" && "$value" != "no" && "$value" != "off" ]]
}

_add_service_if_enabled() {
  local env_name="$1"
  local service_name="$2"
  local value="${!env_name:-}"
  value="${value:-$(_read_env_var "$PROJECT_DIR/.env" "$env_name" || true)}"
  if _enabled "$value"; then
    COMPOSE_SERVICES="${COMPOSE_SERVICES:+$COMPOSE_SERVICES }$service_name"
  fi
}

if [[ -z "${COMPOSE_SERVICES+x}" ]]; then
  COMPOSE_SERVICES=""
  _add_service_if_enabled SERVICE_ORCHESTRATOR orchestrator
  _add_service_if_enabled SERVICE_CAR_CONTROL car_control
  _add_service_if_enabled SERVICE_CAR_MANUAL car_manual
  _add_service_if_enabled SERVICE_FRONTEND frontend
  _add_service_if_enabled SERVICE_VOICE_PROCESSING voice_processing
  _add_service_if_enabled SERVICE_NAVIGATION navigation
  _add_service_if_enabled SERVICE_INFOTAINMENT infotainment
  _add_service_if_enabled SERVICE_CLOUD cloud
fi

echo "[INFO] Waiting 10 seconds before starting orchestration..."
sleep 10

cd "$PROJECT_DIR"

echo "[INFO] Project: $PROJECT_DIR"
echo "[INFO] COMPOSE_PROJECT: $COMPOSE_PROJECT"
echo "[INFO] COMPOSE_SERVICES: ${COMPOSE_SERVICES:-<empty>}"
echo "[INFO] TMUX_SESSION: $TMUX_SESSION"

if [[ -z "${COMPOSE_SERVICES:-}" ]]; then
  echo "[ERROR] COMPOSE_SERVICES is empty. Enable services in .env with SERVICE_*=1 or set COMPOSE_SERVICES manually."
  exit 1
fi

echo "[INFO] Reading optional env vars from .env (if present)..."
AUDIO_MODE="${AUDIO_MODE:-$(_read_env_var "$PROJECT_DIR/.env" AUDIO_MODE || true)}"
AUDIO_READY_TIMEOUT="${AUDIO_READY_TIMEOUT:-$(_read_env_var "$PROJECT_DIR/.env" AUDIO_READY_TIMEOUT || true)}"
SPEAKER_MAC_ADDRESS="${SPEAKER_MAC_ADDRESS:-$(_read_env_var "$PROJECT_DIR/.env" SPEAKER_MAC_ADDRESS || true)}"
BLUETOOTH_AUDIO_PROFILE="${BLUETOOTH_AUDIO_PROFILE:-$(_read_env_var "$PROJECT_DIR/.env" BLUETOOTH_AUDIO_PROFILE || true)}"
SUDO_PWD="${SUDO_PWD:-$(_read_env_var "$PROJECT_DIR/.env" SUDO_PWD || true)}"
ORCH_AUDIT_ENABLED="${ORCH_AUDIT_ENABLED:-$(_read_env_var "$PROJECT_DIR/.env" ORCH_AUDIT_ENABLED || true)}"
ORCH_AUDIT_LOG_ROOT="${ORCH_AUDIT_LOG_ROOT:-$(_read_env_var "$PROJECT_DIR/.env" ORCH_AUDIT_LOG_ROOT || true)}"
MIC_PULSE_SOURCE="${MIC_PULSE_SOURCE:-$(_read_env_var "$PROJECT_DIR/.env" MIC_PULSE_SOURCE || true)}"
MIC_PULSE_SOURCE_PREFIX="${MIC_PULSE_SOURCE_PREFIX:-$(_read_env_var "$PROJECT_DIR/.env" MIC_PULSE_SOURCE_PREFIX || true)}"
MIC_DEVICE_NAME="${MIC_DEVICE_NAME:-$(_read_env_var "$PROJECT_DIR/.env" MIC_DEVICE_NAME || true)}"
PULSE_SOURCE="${PULSE_SOURCE:-$(_read_env_var "$PROJECT_DIR/.env" PULSE_SOURCE || true)}"
PULSE_SINK="${PULSE_SINK:-$(_read_env_var "$PROJECT_DIR/.env" PULSE_SINK || true)}"

: "${AUDIO_MODE:=bluetooth}"
: "${AUDIO_READY_TIMEOUT:=120}"
: "${ORCH_AUDIT_ENABLED:=1}"
: "${ORCH_AUDIT_LOG_ROOT:=$PROJECT_DIR/logs/audit}"
: "${BLUETOOTH_AUDIO_PROFILE:=a2dp}"
: "${MIC_PULSE_SOURCE:=alsa_input.usb-Shenzhen_jiayz_photo_industrial_Ltd_WXMH_mini_E51E621939684C5350344A333033FFl-00.stereo-fallback}"
: "${MIC_PULSE_SOURCE_PREFIX:=alsa_input.usb-Shenzhen_jiayz_photo_industrial_Ltd_WXMH_mini_E51E621939684C5350344A333033FFl-00}"
: "${MIC_DEVICE_NAME:=pulse}"

AUDIO_MODE="${AUDIO_MODE,,}"
case "$AUDIO_MODE" in
  usb|bluetooth)
    ;;
  *)
    echo "[ERROR] Invalid AUDIO_MODE='$AUDIO_MODE'. Use 'usb' or 'bluetooth'."
    exit 1
    ;;
esac

if [[ ! "$AUDIO_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] Invalid AUDIO_READY_TIMEOUT='$AUDIO_READY_TIMEOUT'. Use a positive integer number of seconds."
  exit 1
fi

if [[ "$AUDIO_MODE" == "bluetooth" ]]; then
  BLUETOOTH_AUDIO_PROFILE="${BLUETOOTH_AUDIO_PROFILE,,}"
  case "$BLUETOOTH_AUDIO_PROFILE" in
    a2dp)
      ;;
    hfu|hfp|handsfree|handsfree_head_unit)
      BLUETOOTH_AUDIO_PROFILE="hfu"
      ;;
    *)
      echo "[ERROR] Invalid BLUETOOTH_AUDIO_PROFILE='$BLUETOOTH_AUDIO_PROFILE'. Use 'a2dp' or 'hfu'."
      exit 1
      ;;
  esac
else
  if [[ -z "${PULSE_SINK:-}" ]]; then
    echo "[ERROR] PULSE_SINK is required when AUDIO_MODE=usb."
    exit 1
  fi
  if [[ -z "${PULSE_SOURCE:-}" ]]; then
    echo "[ERROR] PULSE_SOURCE is required when AUDIO_MODE=usb."
    exit 1
  fi
  if [[ "$PULSE_SOURCE" == *.monitor ]]; then
    echo "[ERROR] PULSE_SOURCE='$PULSE_SOURCE' is a monitor source, not a microphone."
    exit 1
  fi
fi

echo "[INFO] AUDIO_MODE: $AUDIO_MODE"
echo "[INFO] AUDIO_READY_TIMEOUT: ${AUDIO_READY_TIMEOUT}s"
echo "[INFO] SPEAKER_MAC_ADDRESS: ${SPEAKER_MAC_ADDRESS:-<empty>}"
echo "[INFO] BLUETOOTH_AUDIO_PROFILE: $BLUETOOTH_AUDIO_PROFILE"
echo "[INFO] SUDO_PWD: ${SUDO_PWD:+<set>}"
echo "[INFO] ORCH_AUDIT_ENABLED: $ORCH_AUDIT_ENABLED"
echo "[INFO] MIC_PULSE_SOURCE: $MIC_PULSE_SOURCE"
echo "[INFO] MIC_PULSE_SOURCE_PREFIX: $MIC_PULSE_SOURCE_PREFIX"

MAC_PA="${SPEAKER_MAC_ADDRESS//:/_}"
MAC_PA="${MAC_PA^^}"
BT_A2DP_SINK="bluez_sink.${MAC_PA}.a2dp_sink"
BT_HFU_SINK="bluez_sink.${MAC_PA}.handsfree_head_unit"
BT_HFU_SOURCE="bluez_source.${MAC_PA}.handsfree_head_unit"
BLUETOOTH_READY_FILE="${XDG_RUNTIME_DIR:-/tmp}/orchestrator-on-edge-${UID}-bluetooth.ready"

if [[ "$AUDIO_MODE" == "bluetooth" ]]; then
  case "$BLUETOOTH_AUDIO_PROFILE" in
    a2dp) BT_AUDIO_SINK="$BT_A2DP_SINK" ;;
    hfu) BT_AUDIO_SINK="$BT_HFU_SINK" ;;
  esac

  if [[ -z "${PULSE_SOURCE:-}" || "$PULSE_SOURCE" == "pulse" ]]; then
    case "$BLUETOOTH_AUDIO_PROFILE" in
      a2dp) PULSE_SOURCE="$MIC_PULSE_SOURCE" ;;
      hfu) PULSE_SOURCE="$BT_HFU_SOURCE" ;;
    esac
  fi
  : "${PULSE_SINK:=$BT_AUDIO_SINK}"
fi
export AUDIO_MODE AUDIO_READY_TIMEOUT BLUETOOTH_AUDIO_PROFILE BLUETOOTH_READY_FILE MIC_DEVICE_NAME PULSE_SOURCE PULSE_SINK

echo "[INFO] PULSE_SINK: $PULSE_SINK"
echo "[INFO] PULSE_SOURCE: $PULSE_SOURCE"

_wait_bluetooth_audio_ready() {
  local timeout="${1:-120}"
  local start now info default_sink default_source

  start="$(date +%s)"
  while true; do
    info="$(pactl info 2>/dev/null || true)"
    default_sink="$(sed -n 's/^Default Sink: //p' <<< "$info" | tail -n 1)"
    default_source="$(sed -n 's/^Default Source: //p' <<< "$info" | tail -n 1)"

    if [[ -f "$BLUETOOTH_READY_FILE" ]] \
      && [[ "$(<"$BLUETOOTH_READY_FILE")" == "$BLUETOOTH_AUDIO_PROFILE" ]] \
      && [[ "$default_sink" == "$BT_AUDIO_SINK" ]]; then
      echo "[INFO] Bluetooth $BLUETOOTH_AUDIO_PROFILE ready: sink=$default_sink source=$default_source"
      return 0
    elif [[ "$default_sink" == "$BT_AUDIO_SINK" && -n "$default_sink" ]]; then
      # Fallback: sink is already active and set to the expected bluetooth profile
      echo "[INFO] Bluetooth $BLUETOOTH_AUDIO_PROFILE ready (sink confirmed): sink=$default_sink source=$default_source"
      return 0
    fi

    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "[ERROR] Bluetooth $BLUETOOTH_AUDIO_PROFILE not ready after ${timeout}s"
      echo "[ERROR] Current sink=${default_sink:-<empty>} source=${default_source:-<empty>}"
      echo "[ERROR] Expected sink=$BT_AUDIO_SINK"
      return 1
    fi

    sleep 2
  done
}

_resolve_pulse_source() {
  local requested="$1"
  local prefix="$2"
  local sources

  sources="$(pactl list sources short 2>/dev/null | awk '{print $2}' || true)"

  if [[ -n "$requested" ]] && grep -Fxq "$requested" <<< "$sources"; then
    printf '%s' "$requested"
    return 0
  fi

  if [[ -n "$prefix" ]]; then
    awk -v prefix="$prefix" 'index($0, prefix) == 1 { print; exit }' <<< "$sources"
    return 0
  fi

  if [[ -z "$requested" ]]; then
    awk 'index($0, "alsa_input.usb-") == 1 { print; exit }' <<< "$sources"
  fi
}

_wait_pulse_source_ready() {
  local requested="$1"
  local prefix="$2"
  local timeout="${3:-120}"
  local start now info default_source resolved_source

  start="$(date +%s)"
  while true; do
    resolved_source="$(_resolve_pulse_source "$requested" "$prefix")"
    if [[ -n "$resolved_source" ]]; then
      pactl set-default-source "$resolved_source" || true
      info="$(pactl info 2>/dev/null || true)"
      default_source="$(sed -n 's/^Default Source: //p' <<< "$info" | tail -n 1)"

      if [[ "$default_source" == "$resolved_source" ]]; then
        PULSE_SOURCE="$resolved_source"
        export PULSE_SOURCE
        echo "[INFO] Pulse source ready: source=$default_source"
        return 0
      fi
    fi

    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "[ERROR] Pulse source not ready after ${timeout}s"
      echo "[ERROR] Expected source=$requested or prefix=$prefix"
      echo "[ERROR] Current sources:"
      pactl list sources short 2>/dev/null | sed 's/^/[ERROR]   /' || true
      return 1
    fi

    sleep 2
  done
}

_wait_usb_audio_ready() {
  local requested_sink="$1"
  local requested_source="$2"
  local timeout="${3:-120}"
  local start now info default_sink default_source sinks sources

  start="$(date +%s)"
  while true; do
    sinks="$(pactl list sinks short 2>/dev/null | awk '{print $2}' || true)"
    sources="$(pactl list sources short 2>/dev/null | awk '{print $2}' || true)"

    if grep -Fxq "$requested_sink" <<< "$sinks" \
      && grep -Fxq "$requested_source" <<< "$sources"; then
      pactl set-default-sink "$requested_sink"
      pactl set-default-source "$requested_source"
      pactl set-sink-mute "$requested_sink" 0 || true
      pactl set-source-mute "$requested_source" 0 || true

      info="$(pactl info 2>/dev/null || true)"
      default_sink="$(sed -n 's/^Default Sink: //p' <<< "$info" | tail -n 1)"
      default_source="$(sed -n 's/^Default Source: //p' <<< "$info" | tail -n 1)"
      if [[ "$default_sink" == "$requested_sink" && "$default_source" == "$requested_source" ]]; then
        echo "[INFO] USB audio ready: sink=$default_sink source=$default_source"
        return 0
      fi
    fi

    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "[ERROR] USB audio not ready after ${timeout}s"
      echo "[ERROR] Expected sink=$requested_sink"
      echo "[ERROR] Expected source=$requested_source"
      echo "[ERROR] Current sinks:"
      pactl list sinks short 2>/dev/null | sed 's/^/[ERROR]   /' || true
      echo "[ERROR] Current sources:"
      pactl list sources short 2>/dev/null | sed 's/^/[ERROR]   /' || true
      return 1
    fi

    sleep 2
  done
}

echo "[INFO] Checking existing tmux session '$TMUX_SESSION'..."
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "[INFO] Existing tmux session found, killing it..."
  tmux kill-session -t "$TMUX_SESSION" || true
fi

echo "[INFO] Creating new tmux session '$TMUX_SESSION'..."

tmux new-session -d -s "$TMUX_SESSION" -n "llama"
if [[ "$ORCH_AUDIT_ENABLED" != "0" && "$ORCH_AUDIT_ENABLED" != "false" && "$ORCH_AUDIT_ENABLED" != "FALSE" ]]; then
  echo "[INFO] Creating 'audit' window..."
  echo "[INFO] ORCH_AUDIT_LOG_ROOT: $ORCH_AUDIT_LOG_ROOT"
  tmux new-window -t "$TMUX_SESSION" -n "audit" "cd \"$PROJECT_DIR\" && PROJECT_DIR=\"$PROJECT_DIR\" COMPOSE_PROJECT=\"$COMPOSE_PROJECT\" ORCH_AUDIT_LOG_ROOT=\"$ORCH_AUDIT_LOG_ROOT\" python3 ./scripts/audit_watch.py"
else
  echo "[INFO] ORCH_AUDIT_ENABLED=$ORCH_AUDIT_ENABLED; skipping 'audit' window."
fi
MODEL_DIR="${MODEL_DIR:-$(_read_env_var "$PROJECT_DIR/.env" LLM_MODEL_DIR || true)}"
MODEL_FILE="${MODEL_FILE:-$(_read_env_var "$PROJECT_DIR/.env" LLM_MODEL_FILE || true)}"
LLAMA_PORT="${LLAMA_PORT:-$(_read_env_var "$PROJECT_DIR/.env" LLM_PORT || true)}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$(_read_env_var "$PROJECT_DIR/.env" LLAMA_SERVER_BIN || true)}"

: "${MODEL_DIR:=$PROJECT_DIR/models}"
: "${MODEL_FILE:=qwen3_sft_merged.Q4_K_M.gguf}"
: "${LLAMA_PORT:=8080}"
: "${LLAMA_SERVER_BIN:=/opt/llama.cpp/bin/llama-server}"

echo "[INFO] LLM MODEL_DIR: $MODEL_DIR"
echo "[INFO] LLM MODEL_FILE: $MODEL_FILE"
echo "[INFO] LLM PORT: $LLAMA_PORT"
echo "[INFO] LLM SERVER BIN: $LLAMA_SERVER_BIN"

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "[ERROR] Native llama-server is missing or not executable: $LLAMA_SERVER_BIN"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/$MODEL_FILE" ]]; then
  echo "[ERROR] LLM model is missing: $MODEL_DIR/$MODEL_FILE"
  exit 1
fi

echo "[INFO] Starting llama-server in tmux window 'llama'..."
tmux send-keys -t "$TMUX_SESSION:llama" "exec \"$LLAMA_SERVER_BIN\" -m \"$MODEL_DIR/$MODEL_FILE\" --jinja --port \"$LLAMA_PORT\" --host 0.0.0.0 -ngl 99 -t 12 -fa --log-colors --parallel 1 --n-predict 1024 --ctx-size 4096 --batch-size 512" C-m

echo "[INFO] Creating 'warmup' window..."
tmux new-window -t "$TMUX_SESSION" -n "warmup"
tmux send-keys -t "$TMUX_SESSION:warmup" "sleep 10; curl -s -X POST \"http://localhost:${LLAMA_PORT:-8080}/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{ \"model\": \"llama\", \"messages\": [ {\"role\":\"system\",\"content\":\"You are a helpful assistant.\"}, {\"role\":\"user\",\"content\":\"Say hello briefly.\"} ] }' >/dev/null || true; curl -s -X POST \"http://localhost:${LLAMA_PORT:-8080}/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{ \"model\": \"llama\", \"messages\": [ {\"role\":\"system\",\"content\":\"You are a helpful assistant.\"}, {\"role\":\"user\",\"content\":\"What is 2 plus 2?\"} ] }' >/dev/null || true; curl -s -X POST \"http://localhost:${LLAMA_PORT:-8080}/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{ \"model\": \"llama\", \"messages\": [ {\"role\":\"system\",\"content\":\"You are a helpful assistant.\"}, {\"role\":\"user\",\"content\":\"Give me a very short fun fact.\"} ] }' >/dev/null || true; echo \"[DONE] warmup\"; exec bash" C-m

if [[ "$AUDIO_MODE" == "bluetooth" ]]; then
  rm -f -- "$BLUETOOTH_READY_FILE"
  echo "[INFO] Creating 'bluetooth' window (profile: $BLUETOOTH_AUDIO_PROFILE)..."
  tmux new-window -t "$TMUX_SESSION" -n "bluetooth"
  tmux send-keys -t "$TMUX_SESSION:bluetooth" "cd \"$PROJECT_DIR\" && chmod +x ./setup_blue.sh || true; BLUETOOTH_READY_FILE=\"$BLUETOOTH_READY_FILE\" BLUETOOTH_AUDIO_PROFILE=\"$BLUETOOTH_AUDIO_PROFILE\" ./setup_blue.sh; exec bash" C-m

  echo "[INFO] Waiting for Bluetooth $BLUETOOTH_AUDIO_PROFILE before starting docker compose..."
  _wait_bluetooth_audio_ready "$AUDIO_READY_TIMEOUT"

  echo "[INFO] Waiting for Pulse input source before starting docker compose..."
  if [[ "$BLUETOOTH_AUDIO_PROFILE" == "a2dp" ]]; then
    _wait_pulse_source_ready "$PULSE_SOURCE" "$MIC_PULSE_SOURCE_PREFIX" "$AUDIO_READY_TIMEOUT"
  else
    _wait_pulse_source_ready "$PULSE_SOURCE" "" "$AUDIO_READY_TIMEOUT"
  fi
else
  echo "[INFO] Waiting for configured USB audio before starting docker compose..."
  _wait_usb_audio_ready "$PULSE_SINK" "$PULSE_SOURCE" "$AUDIO_READY_TIMEOUT"
fi

echo "[INFO] Creating 'stack' window and starting docker compose (audio mode: $AUDIO_MODE)..."
tmux new-window -t "$TMUX_SESSION" -n "stack" "cd \"$PROJECT_DIR\" && MIC_DEVICE_NAME=\"$MIC_DEVICE_NAME\" PULSE_SOURCE=\"$PULSE_SOURCE\" PULSE_SINK=\"$PULSE_SINK\" docker compose -p \"$COMPOSE_PROJECT\" up --no-build --pull never --timestamps $COMPOSE_SERVICES"

echo "[DONE] tmux session created. Attach with:"
echo "  tmux attach -t \"$TMUX_SESSION\""

while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
  sleep 5
done
