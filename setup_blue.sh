#!/bin/bash

set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_read_env_var() {
  local env_file="$1"
  local key="$2"
  [[ -f "$env_file" ]] || return 1
  local line
  line="$(sed -nE "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)[[:space:]]*$/\\1/p" "$env_file" | tail -n 1)"
  [[ -n "${line:-}" ]] || return 1
  line="${line%\"}"
  line="${line#\"}"
  line="${line%\'}"
  line="${line#\'}"
  printf '%s' "$line"
}

SPEAKER_MAC_ADDRESS="${SPEAKER_MAC_ADDRESS:-$(_read_env_var "$PROJECT_DIR/.env" SPEAKER_MAC_ADDRESS || true)}"
SPEAKER_MAC_ADDRESS="${SPEAKER_MAC_ADDRESS:-84:D3:52:E8:9D:0F}"
BLUETOOTH_AUDIO_PROFILE="${BLUETOOTH_AUDIO_PROFILE:-$(_read_env_var "$PROJECT_DIR/.env" BLUETOOTH_AUDIO_PROFILE || true)}"
BLUETOOTH_AUDIO_PROFILE="${BLUETOOTH_AUDIO_PROFILE:-a2dp}"
BLUETOOTH_AUDIO_PROFILE="${BLUETOOTH_AUDIO_PROFILE,,}"
SUDO_PWD="${SUDO_PWD:-$(_read_env_var "$PROJECT_DIR/.env" SUDO_PWD || true)}"
MAC_PA="${SPEAKER_MAC_ADDRESS//:/_}"
MAC_PA="${MAC_PA^^}"

_run_sudo() {
  if [[ -n "${SUDO_PWD:-}" ]]; then
    echo "$SUDO_PWD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

_connect_bluetooth() {
  bluetoothctl <<EOF
power on
agent on
default-agent
trust $SPEAKER_MAC_ADDRESS
connect $SPEAKER_MAC_ADDRESS
EOF
}

CARD="bluez_card.${MAC_PA}"
A2DP_SINK="bluez_sink.${MAC_PA}.a2dp_sink"
HFU_SINK="bluez_sink.${MAC_PA}.handsfree_head_unit"
HFU_SOURCE="bluez_source.${MAC_PA}.handsfree_head_unit"

case "$BLUETOOTH_AUDIO_PROFILE" in
  a2dp)
    PA_PROFILE="a2dp_sink"
    TARGET_SINK="$A2DP_SINK"
    PROFILE_LABEL="A2DP stereo"
    ;;
  hfu|hfp|handsfree|handsfree_head_unit)
    BLUETOOTH_AUDIO_PROFILE="hfu"
    PA_PROFILE="handsfree_head_unit"
    TARGET_SINK="$HFU_SINK"
    PROFILE_LABEL="HFU/HFP handsfree"
    ;;
  *)
    echo "[ERROR] Invalid BLUETOOTH_AUDIO_PROFILE='$BLUETOOTH_AUDIO_PROFILE'. Use 'a2dp' or 'hfu'."
    exit 1
    ;;
esac

echo "[INFO] SPEAKER_MAC_ADDRESS: $SPEAKER_MAC_ADDRESS (PulseAudio: $MAC_PA)"
echo "[INFO] BLUETOOTH_AUDIO_PROFILE: $BLUETOOTH_AUDIO_PROFILE ($PROFILE_LABEL)"
echo "[INFO] Checking Bluetooth block state..."
if rfkill list bluetooth | grep -q "Soft blocked: yes"; then
  echo "[INFO] Bluetooth is blocked. Unblocking..."
  _run_sudo rfkill unblock bluetooth
  sleep 2
fi

echo "[INFO] Restarting PulseAudio and Bluetooth..."
if pactl info 2>/dev/null | grep -q "Server Name:.*PipeWire" || systemctl --user is-active --quiet pipewire-pulse.service; then
  systemctl --user restart wireplumber.service pipewire-pulse.service pipewire.service
else
  pulseaudio -k || true
  systemctl --user restart pulseaudio
fi
_run_sudo systemctl restart bluetooth
sleep 5

echo "[INFO] Connecting to Bluetooth speaker $SPEAKER_MAC_ADDRESS..."
_connect_bluetooth || true
sleep 5

echo "[INFO] Please PRESS the Bluetooth button on the speaker if it is not already connected"
echo "[INFO] Waiting for Bluetooth $PROFILE_LABEL audio sink..."

### ---- MAIN LOOP ----
CONNECT_ATTEMPTS=0
while true; do
  CARDS="$(pactl list cards short)"
  SINKS="$(pactl list sinks short)"
  SOURCES="$(pactl list sources short)"

  if ! echo "$CARDS" | grep -q "$CARD"; then
    CONNECT_ATTEMPTS=$((CONNECT_ATTEMPTS + 1))
    if (( CONNECT_ATTEMPTS % 3 == 1 )); then
      echo "[INFO] Bluetooth card not visible yet -> retrying connect..."
      _connect_bluetooth || true
    fi
  fi

  if echo "$CARDS" | grep -q "$CARD" && ! echo "$SINKS" | grep -q "$TARGET_SINK"; then
    echo "[INFO] Bluetooth card detected -> switching to $PROFILE_LABEL profile, waiting 4s..."
    pactl set-card-profile "$CARD" "$PA_PROFILE" || true
    sleep 4
    continue
  fi

  if echo "$SINKS" | grep -q "$TARGET_SINK"; then
    echo "[INFO] $PROFILE_LABEL sink detected"

    pactl set-card-profile "$CARD" "$PA_PROFILE" || true
    pactl set-default-sink "$TARGET_SINK" || true

    echo "[INFO] Setting speaker volume to 100%"
    pactl set-sink-volume "$TARGET_SINK" 100% || true

    if [[ "$BLUETOOTH_AUDIO_PROFILE" == "hfu" ]] && echo "$SOURCES" | grep -q "$HFU_SOURCE"; then
      pactl set-source-volume "$HFU_SOURCE" 100% || true
    fi

    # run_all.sh selects the configured USB microphone after output is ready.
    if pactl info | grep -q "^Default Sink: $TARGET_SINK$"; then
      if [[ -n "${BLUETOOTH_READY_FILE:-}" ]]; then
        printf '%s\n' "$BLUETOOTH_AUDIO_PROFILE" > "$BLUETOOTH_READY_FILE"
      fi
      echo "[DONE] Device connected and configured"
      echo "Output: $PROFILE_LABEL speaker | Input: unchanged default source"
      pactl info | grep -E "Default Sink|Default Source"
      break
    fi
  fi

  sleep 4
done
