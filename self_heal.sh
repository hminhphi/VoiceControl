#!/usr/bin/env bash
set -Eeuo pipefail

LOG_FILE="${SELF_HEAL_LOG_FILE:-/var/log/orchestrator-on-edge-self-heal.log}"
ROOT_MIN_AVAIL_MB="${ROOT_MIN_AVAIL_MB:-2048}"
ROOT_CRITICAL_USE_PCT="${ROOT_CRITICAL_USE_PCT:-95}"
SELF_HEAL_USER="${SELF_HEAL_USER:-nvidia}"
SELF_HEAL_UID="${SELF_HEAL_UID:-1000}"
TMUX_TMP_DIR="${TMUX_TMP_DIR:-/tmp/tmux-${SELF_HEAL_UID}}"
JOURNAL_VACUUM_SIZE="${JOURNAL_VACUUM_SIZE:-200M}"

_timestamp() {
  date "+%Y-%m-%d %H:%M:%S%z"
}

_log() {
  local message="$*"
  local line
  line="[$(_timestamp)] $message"

  if mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null; then
    printf '%s\n' "$line" | tee -a "$LOG_FILE" >/dev/null || printf '%s\n' "$line"
  else
    printf '%s\n' "$line"
  fi
}

_run_allow_fail() {
  local status
  _log "+ $*"
  set +e
  "$@" 2>&1 | while IFS= read -r line; do
    _log "  $line"
  done
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    _log "WARN: command exited with status $status: $*"
  fi
}

_root_disk_stats() {
  df -Pm / | awk 'NR == 2 { gsub("%", "", $5); print $4, $5 }'
}

_cleanup_system_logs() {
  _log "Cleaning large system logs because root disk is critically full"

  _run_allow_fail journalctl --vacuum-size="$JOURNAL_VACUUM_SIZE"

  for log_file in /var/log/syslog /var/log/kern.log; do
    if [[ -f "$log_file" ]]; then
      _log "Truncating active log: $log_file"
      _run_allow_fail truncate -s 0 "$log_file"
    fi
  done

  local rotated_logs=()
  shopt -s nullglob
  rotated_logs+=(/var/log/syslog.*)
  rotated_logs+=(/var/log/kern.log.*)
  shopt -u nullglob

  if [[ "${#rotated_logs[@]}" -gt 0 ]]; then
    _log "Removing rotated syslog/kern.log files: ${#rotated_logs[@]} file(s)"
    _run_allow_fail rm -f -- "${rotated_logs[@]}"
  else
    _log "No rotated syslog/kern.log files found"
  fi
}

_cleanup_stale_tmux_dir() {
  if [[ ! -e "$TMUX_TMP_DIR" ]]; then
    _log "Tmux temp dir does not exist: $TMUX_TMP_DIR"
    return
  fi

  if pgrep -u "$SELF_HEAL_USER" tmux >/dev/null 2>&1; then
    _log "Tmux process exists for user $SELF_HEAL_USER; keeping $TMUX_TMP_DIR"
    return
  fi

  _log "No tmux process found for user $SELF_HEAL_USER; removing stale $TMUX_TMP_DIR"
  _run_allow_fail rm -rf -- "$TMUX_TMP_DIR"
}

main() {
  local avail_mb
  local used_pct

  _log "Starting orchestrator-on-edge startup self-heal"
  _log "Thresholds: root used >= ${ROOT_CRITICAL_USE_PCT}% or available < ${ROOT_MIN_AVAIL_MB}MB"

  read -r avail_mb used_pct < <(_root_disk_stats)
  _log "Root disk before cleanup: used=${used_pct}% available=${avail_mb}MB"

  if [[ "$used_pct" -ge "$ROOT_CRITICAL_USE_PCT" || "$avail_mb" -lt "$ROOT_MIN_AVAIL_MB" ]]; then
    _cleanup_system_logs
    read -r avail_mb used_pct < <(_root_disk_stats)
    _log "Root disk after log cleanup: used=${used_pct}% available=${avail_mb}MB"
  else
    _log "Root disk is below cleanup threshold"
  fi

  _cleanup_stale_tmux_dir

  read -r avail_mb used_pct < <(_root_disk_stats)
  if [[ "$used_pct" -ge 99 || "$avail_mb" -lt 128 ]]; then
    _log "ERROR: root disk is still too full for reliable startup: used=${used_pct}% available=${avail_mb}MB"
    exit 1
  fi

  _log "Startup self-heal complete"
}

main "$@"
