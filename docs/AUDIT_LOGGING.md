# Audit logging

This repo includes a host-side watcher for runtime audit logs:

- Docker container lifecycle changes from `docker events`, including `die`, `oom`, `kill`, restart, health changes, exit code, OOM flag, start/finish timestamps, and inspect state.
- Docker stdout/stderr lines from compose containers as `container_log` events. This includes `voice_processing`.
- PulseAudio/PipeWire Pulse compatibility changes from `pactl subscribe`, with `pactl info`, default sink/source, mute state, and volume.
- Periodic host snapshots: disk free/used space, CPU, RAM, top processes, Docker stats, and GPU data from `tegrastats` on Jetson or `nvidia-smi` when available.
- Newline-delimited JSON with local timezone timestamps.
- Built-in size rotation, so logs do not grow without bound.

## Default location

`run_all.sh` starts the watcher in a tmux window named `audit`.

Default log file:

```bash
/home/$USER/orchestrator-audit/logs/audit.jsonl
```

Rotated files:

```bash
/home/$USER/orchestrator-audit/logs/audit.jsonl.1
/home/$USER/orchestrator-audit/logs/audit.jsonl.2
```

## Configuration

Set these environment variables in `.env` or before running `run_all.sh`:

```bash
ORCH_AUDIT_LOG_ROOT=/home/hpc/orchestrator-audit/logs
ORCH_AUDIT_MAX_BYTES=25000000
ORCH_AUDIT_BACKUP_COUNT=20
ORCH_AUDIT_TELEMETRY_INTERVAL=15
ORCH_AUDIT_COMPOSE_INTERVAL=30
ORCH_AUDIT_DOCKER_LOG_TAIL=20
DOCKER_LOG_MAX_SIZE=10m
DOCKER_LOG_MAX_FILE=5
```

With defaults, the audit files keep about `25 MB * 21` of recent JSONL logs before old rotated files are replaced.
Docker's own `json-file` logs are also capped by `DOCKER_LOG_MAX_SIZE` and `DOCKER_LOG_MAX_FILE` in `docker-compose.yml`.

## Read useful events

Container exits:

```bash
grep '"event": "container_event"' /home/hpc/orchestrator-audit/logs/audit.jsonl | grep '"action": "die"'
```

`voice_processing` stdout/stderr:

```bash
grep '"event": "container_log"' /home/hpc/orchestrator-audit/logs/audit.jsonl | grep '"compose_service": "voice_processing"'
```

Audio default sink/source or mute changes:

```bash
grep '"event": "pulse_change"' /home/hpc/orchestrator-audit/logs/audit.jsonl
```

Disk and peak CPU/GPU snapshots:

```bash
grep '"event": "system_snapshot"' /home/hpc/orchestrator-audit/logs/audit.jsonl
```

## Optional OS logrotate

The Python watcher already rotates by size. If you also want compression and age-based deletion, this is the matching `/etc/logrotate.d/orchestrator-audit` content:

```conf
/home/hpc/orchestrator-audit/logs/*.jsonl* {
    daily
    rotate 14
    size 25M
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

Use `copytruncate` because the watcher keeps the active file open.
