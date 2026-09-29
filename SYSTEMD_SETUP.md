## Systemd Autostart Setup for Orchestrator on Edge

This document describes how to configure the Jetson (Ubuntu with systemd) to automatically start the orchestrator stack on boot using `run_all.sh` and a `systemd` service.

### Prerequisites

- Jetson board running Ubuntu with systemd (e.g., Jetson Orin AGX).
- Project checked out at:
  - `/mnt/nvme/Projects/orchestrator-on-edge`
- Docker and `docker compose` already installed and working.
- `tmux` installed.

### Script Entry Point

The main entry point used by systemd is:

- `run_all.sh`

This script:

- Creates a tmux session (default name: `orch_edge`).
- Starts:
  - A `llama-server` container via `jetson-containers`.
  - The main Docker stack via `docker compose`.
  - The configured USB or Bluetooth audio setup.
- Keeps running while the tmux session exists so that systemd does not kill the tmux processes.

### Audio mode configuration

`run_all.sh` supports two audio modes selected through the project `.env` file:

- `AUDIO_MODE=usb`: wait for an explicitly configured PulseAudio/PipeWire sink and microphone source. Bluetooth setup is skipped.
- `AUDIO_MODE=bluetooth`: retain the existing A2DP/HFU setup through `setup_blue.sh`.

The default is `bluetooth` for backward compatibility. `AUDIO_READY_TIMEOUT` controls how long startup waits for audio. If the device is still unavailable after this timeout, `run_all.sh` exits with an error and systemd retries according to `Restart=` and `RestartSec=`.

#### Jabra Speak 410 USB

Use the following values in `/home/nvidia/fa_ai/orchestrator-on-edge/.env`:

```dotenv
AUDIO_MODE=usb
AUDIO_READY_TIMEOUT=120
MIC_DEVICE_NAME=pulse

PULSE_SINK=alsa_output.usb-0b0e_Jabra_SPEAK_410_USB_08C8C2A7FCA0x011200-00.analog-stereo
PULSE_SOURCE=alsa_input.usb-0b0e_Jabra_SPEAK_410_USB_08C8C2A7FCA0x011200-00.mono-fallback
```

`PULSE_SOURCE` must be the real microphone source. Do not select a source ending in `.monitor`; monitor sources record the speaker playback rather than the microphone.

The Bluetooth variables may remain in `.env`; they are ignored while `AUDIO_MODE=usb`:

```dotenv
SPEAKER_MAC_ADDRESS=84:D3:52:E8:9D:0F
BLUETOOTH_AUDIO_PROFILE=hfu
```

If the USB device name changes, reconnect or power on the device and find the exact names with:

```bash
pactl list short sinks
pactl list short sources
```

Choose the Jabra entry from `sinks` for `PULSE_SINK` and the `alsa_input...Jabra...` entry from `sources` for `PULSE_SOURCE`.

Restart and follow startup logs:

```bash
sudo systemctl restart orchestrator-on-edge.service
journalctl -u orchestrator-on-edge.service -f
```

Verify host routing and the environment passed to `voice_processing`:

```bash
pactl info | grep -E 'Default Sink|Default Source'

docker inspect voice_processing --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E 'PULSE_(SINK|SOURCE)'

docker logs voice_processing 2>&1 \
  | grep -E 'active record source|active playback sink'
```

Both source and sink should contain `Jabra_SPEAK_410_USB`.

#### Switch back to Bluetooth

Set `AUDIO_MODE=bluetooth` and choose the existing Bluetooth profile:

```dotenv
AUDIO_MODE=bluetooth
AUDIO_READY_TIMEOUT=120
SPEAKER_MAC_ADDRESS=84:D3:52:E8:9D:0F
BLUETOOTH_AUDIO_PROFILE=hfu
```

Remove or comment out the USB-specific `PULSE_SINK` and `PULSE_SOURCE` lines when switching back. This lets `run_all.sh` derive the correct Bluetooth endpoints from the MAC address and profile:

```dotenv
# PULSE_SINK=alsa_output.usb-...
# PULSE_SOURCE=alsa_input.usb-...
```

For high-quality output without a Bluetooth microphone, use `BLUETOOTH_AUDIO_PROFILE=a2dp` and configure the separate USB microphone through the existing `MIC_PULSE_SOURCE` and `MIC_PULSE_SOURCE_PREFIX` variables. Restart `orchestrator-on-edge.service` after changing `.env`.

### Anker PowerConf S3 Handsfree Autostart

For stable microphone input on Jetson, run the Anker PowerConf S3 as a separate user service. This keeps reconnecting the speaker and forces PulseAudio to use the Handsfree Unit profile instead of falling back to A2DP output plus analog input.

The tracked files are:

- `scripts/anker-powerconf-auto.sh`
- `scripts/anker-powerconf-auto.service`

Install them for user `hpc`:

```bash
mkdir -p /home/hpc/.local/bin /home/hpc/.config/systemd/user
cp /mnt/nvme/Projects/orchestrator-on-edge/scripts/anker-powerconf-auto.sh /home/hpc/.local/bin/anker-powerconf-auto.sh
cp /mnt/nvme/Projects/orchestrator-on-edge/scripts/anker-powerconf-auto.service /home/hpc/.config/systemd/user/anker-powerconf-auto.service
chmod +x /home/hpc/.local/bin/anker-powerconf-auto.sh
systemctl --user daemon-reload
systemctl --user enable --now anker-powerconf-auto.service
sudo loginctl enable-linger hpc
```

The service reads `SPEAKER_MAC_ADDRESS` from `/mnt/nvme/Projects/orchestrator-on-edge/.env` when present. If it is not set, it defaults to:

```bash
SPEAKER_MAC_ADDRESS=84:D3:52:E8:9D:0F
```

Verify after the speaker is powered on:

```bash
systemctl --user status anker-powerconf-auto.service
pactl info | grep -E "Default Sink|Default Source"
```

Expected PulseAudio defaults:

```text
Default Sink: bluez_sink.84_D3_52_E8_9D_0F.handsfree_head_unit
Default Source: bluez_source.84_D3_52_E8_9D_0F.handsfree_head_unit
```

If your PulseAudio build exposes the headset profile with the alternate name, the script falls back to:

```text
bluez_sink.84_D3_52_E8_9D_0F.headset_head_unit
bluez_source.84_D3_52_E8_9D_0F.headset_head_unit
```

`setup_blue.sh` can still be used as a manual recovery helper, but boot-time Bluetooth reconnect should be owned by `anker-powerconf-auto.service`.

### Service File

Service file path on the system:

- `/etc/systemd/system/orchestrator-on-edge.service`

Recommended contents:

```ini
[Unit]
Description=Orchestrator-on-edge (run_all.sh)
Wants=network-online.target docker.service bluetooth.service
After=network-online.target docker.service bluetooth.service

[Service]
Type=simple
User=hpc
WorkingDirectory=/mnt/nvme/Projects/orchestrator-on-edge
EnvironmentFile=-/mnt/nvme/Projects/orchestrator-on-edge/.env
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
ExecStartPre=/bin/sleep 30
ExecStart=/mnt/nvme/Projects/orchestrator-on-edge/run_all.sh
StandardOutput=append:/var/log/orchestrator-on-edge.log
StandardError=append:/var/log/orchestrator-on-edge.log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Notes

- **User and UID**: If your username is not `hpc`, change `User=` in the service file. If your UID is not 1000, replace `1000` in `XDG_RUNTIME_DIR` and `DBUS_SESSION_BUS_ADDRESS` with your UID (run `id -u` to get it).
- **Log file**: The service appends to `/var/log/orchestrator-on-edge.log`. Ensure the file exists and is writable by user `hpc`, e.g. `sudo touch /var/log/orchestrator-on-edge.log && sudo chown hpc:hpc /var/log/orchestrator-on-edge.log`.

### Initial Setup

From a terminal:

```bash
sudo cp /mnt/nvme/Projects/orchestrator-on-edge/orchestrator-on-edge.service /etc/systemd/system/orchestrator-on-edge.service
sudo systemctl daemon-reload
sudo systemctl enable orchestrator-on-edge.service
sudo systemctl start orchestrator-on-edge.service
```

### Verifying the Service

Check service status:

```bash
systemctl status orchestrator-on-edge.service
```

Attach to the tmux session created by `run_all.sh`:

```bash
tmux attach -t "orch_edge"
```

If the session does not exist, check logs:

```bash
journalctl -u orchestrator-on-edge.service -e
```

### Stopping and Disabling Autostart

To stop the service and prevent it from starting automatically on boot:

```bash
sudo systemctl stop orchestrator-on-edge.service
sudo systemctl disable orchestrator-on-edge.service
```

To completely remove the service file:

```bash
sudo rm /etc/systemd/system/orchestrator-on-edge.service
sudo systemctl daemon-reload
sudo systemctl reset-failed
```
