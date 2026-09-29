#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
ARTIFACT_DIR="${NIRNAY_ARTIFACT_DIR:-artifacts/phase_a}"
CHECKPOINT="$ARTIFACT_DIR/phase_a.pt"
CHECKPOINT_META="$ARTIFACT_DIR/phase_a.pt.meta.json"
HISTORY_FILE="$ARTIFACT_DIR/history.json"
STATE_FILE="$LOG_DIR/phase_a_supervisor.state"
LOCK_DIR="$LOG_DIR/phase_a_supervisor.lock"
PID_FILE="$LOG_DIR/phase_a.pid"
FAIL_FILE="$LOG_DIR/phase_a_supervisor.failed"

DELAY_SECONDS=0
CANARY_STEPS=250
TOTAL_STEPS=7000
BATCH_SIZE=4
LEARNING_RATE=0.0005
PRETRAINED_LR=""
CONCEPTS_LR=""
CHECKPOINT_EVERY=50
PROBE_EVERY=""
N_SYNTH_HARD=""
MAX_RESTARTS=5
DRY_RUN=0
RUN_PHASE_B=0
RUN_EVAL=1
EVAL_LIMIT=0
DEVICE="${NIRNAY_DEVICE:-mps}"
HF_HOME="${HF_HOME:-/Volumes/KIOXIA 1TB/huggingface_cache}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME HF_HUB_OFFLINE
export PYTHONUNBUFFERED=1
export UV_OFFLINE="${UV_OFFLINE:-1}"

STALL_SECONDS="${NIRNAY_STALL_SECONDS:-1500}"
MIN_FREE_MEM_PCT="${NIRNAY_MIN_FREE_MEM_PCT:-25}"
MIN_FREE_DISK_GB="${NIRNAY_MIN_FREE_DISK_GB:-3}"

usage() {
  printf '%s\n' "Usage: $0 --delay-seconds N [--dry-run] [--with-phase-b]"
  printf '%s\n' "Options: --canary-steps N --total-steps N --batch-size N --lr N --checkpoint-every N --max-restarts N"
  printf '%s\n' "         --pretrained-lr FLOAT --concepts-lr FLOAT (param-group lrs; default = --lr)"
  printf '%s\n' "         --probe-every N (heldout probe cadence; default = checkpoint-every; 0 disables)"
  printf '%s\n' "         --n-synth-hard N (hard-reasoning synthetic items; default 0 = legacy mix)"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --delay-seconds)
      DELAY_SECONDS="${2:-}"
      shift 2
      ;;
    --delay-minutes)
      DELAY_SECONDS=$((${2:-} * 60))
      shift 2
      ;;
    --canary-steps)
      CANARY_STEPS="${2:-}"
      shift 2
      ;;
    --total-steps)
      TOTAL_STEPS="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="${2:-}"
      shift 2
      ;;
    --pretrained-lr)
      PRETRAINED_LR="${2:-}"
      shift 2
      ;;
    --concepts-lr)
      CONCEPTS_LR="${2:-}"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="${2:-}"
      shift 2
      ;;
    --probe-every)
      PROBE_EVERY="${2:-}"
      shift 2
      ;;
    --n-synth-hard)
      N_SYNTH_HARD="${2:-}"
      shift 2
      ;;
    --max-restarts)
      MAX_RESTARTS="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --with-phase-b)
      RUN_PHASE_B=1
      shift
      ;;
    --skip-eval)
      RUN_EVAL=0
      shift
      ;;
    --eval-limit)
      EVAL_LIMIT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

is_uint() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

is_uint "$DELAY_SECONDS" || { printf '%s\n' 'delay must be a non-negative integer' >&2; exit 2; }
is_uint "$CANARY_STEPS" || { printf '%s\n' 'canary steps must be a non-negative integer' >&2; exit 2; }
is_uint "$TOTAL_STEPS" || { printf '%s\n' 'total steps must be a non-negative integer' >&2; exit 2; }
is_uint "$BATCH_SIZE" || { printf '%s\n' 'batch size must be a positive integer' >&2; exit 2; }
is_uint "$CHECKPOINT_EVERY" || { printf '%s\n' 'checkpoint interval must be a non-negative integer' >&2; exit 2; }
if [ -n "$PROBE_EVERY" ]; then
  is_uint "$PROBE_EVERY" || { printf '%s\n' 'probe interval must be a non-negative integer' >&2; exit 2; }
fi
if [ -n "$N_SYNTH_HARD" ]; then
  is_uint "$N_SYNTH_HARD" || { printf '%s\n' 'n-synth-hard must be a non-negative integer' >&2; exit 2; }
fi
is_uint "$MAX_RESTARTS" || { printf '%s\n' 'max restarts must be a non-negative integer' >&2; exit 2; }
is_uint "$EVAL_LIMIT" || { printf '%s\n' 'eval limit must be a non-negative integer' >&2; exit 2; }
is_lr() {
  [ -n "$1" ] || return 1
  printf '%s' "$1" | grep -Eq '^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$'
}
if [ -n "$PRETRAINED_LR" ]; then
  is_lr "$PRETRAINED_LR" || { printf '%s\n' 'pretrained-lr must be a number' >&2; exit 2; }
fi
if [ -n "$CONCEPTS_LR" ]; then
  is_lr "$CONCEPTS_LR" || { printf '%s\n' 'concepts-lr must be a number' >&2; exit 2; }
fi
[ "$CANARY_STEPS" -gt 0 ] || { printf '%s\n' 'canary steps must be positive' >&2; exit 2; }
[ "$TOTAL_STEPS" -ge "$CANARY_STEPS" ] || { printf '%s\n' 'total steps must be at least canary steps' >&2; exit 2; }
[ "$BATCH_SIZE" -ge 2 ] || { printf '%s\n' 'batch size must be at least 2' >&2; exit 2; }
[ -d "$ROOT/data/banking77" ] || { printf '%s\n' 'data/banking77 is missing' >&2; exit 2; }
[ -d "$HF_HOME" ] || { printf 'HF_HOME does not exist: %s\n' "$HF_HOME" >&2; exit 2; }

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'DRY_RUN_OK delay_seconds=%s canary_steps=%s total_steps=%s batch_size=%s lr=%s checkpoint_every=%s max_restarts=%s eval=%s eval_limit=%s phase_b=%s\n' "$DELAY_SECONDS" "$CANARY_STEPS" "$TOTAL_STEPS" "$BATCH_SIZE" "$LEARNING_RATE" "$CHECKPOINT_EVERY" "$MAX_RESTARTS" "$RUN_EVAL" "$EVAL_LIMIT" "$RUN_PHASE_B"
    printf 'WOULD_RUN uv run python -m nirnay.train --phase a --steps %s --batch-size %s --n-synth 512 --banking-dir data/banking77 --out-dir %s --lora-rank 8 --lr %s --seed 13 --checkpoint-every %s --device %s\n' "$TOTAL_STEPS" "$BATCH_SIZE" "$ARTIFACT_DIR" "$LEARNING_RATE" "$CHECKPOINT_EVERY" "$DEVICE"
    if [ -n "$PRETRAINED_LR" ]; then
      printf 'WOULD_PASS --pretrained-lr %s\n' "$PRETRAINED_LR"
    fi
    if [ -n "$CONCEPTS_LR" ]; then
      printf 'WOULD_PASS --concepts-lr %s\n' "$CONCEPTS_LR"
    fi
    if [ "${PROBE_EVERY:-$CHECKPOINT_EVERY}" -gt 0 ]; then
      printf 'WOULD_PASS --probe-every %s --probe-size 256\n' "${PROBE_EVERY:-$CHECKPOINT_EVERY}"
    fi
    if [ -n "$N_SYNTH_HARD" ]; then
      printf 'WOULD_PASS --n-synth-hard %s\n' "$N_SYNTH_HARD"
    fi
  if [ "$RUN_EVAL" -eq 1 ]; then
    printf 'WOULD_RUN uv run python scripts/eval_checkpoint.py --checkpoint %s/phase_a.pt --data-dir data/banking77 --batch-size %s --limit %s --device %s\n' "$ARTIFACT_DIR" "$BATCH_SIZE" "$EVAL_LIMIT" "$DEVICE"
  fi
  exit 0
fi

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"
rm -f "$FAIL_FILE"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

write_state() {
  local temporary="$STATE_FILE.tmp.$$"
  printf '%s\n' "$1" > "$temporary"
  mv -f "$temporary" "$STATE_FILE"
}

read_state() {
  if [ -f "$STATE_FILE" ]; then
    cat "$STATE_FILE"
  else
    printf '%s\n' "unknown"
  fi
}

fail_hard() {
  local message="$1"
  local rc="${2:-1}"
  log "FATAL: $message"
  printf '%s\n' "$message" > "$FAIL_FILE"
  write_state "failed: $message"
  exit "$rc"
}

read_pid() {
  local value=""
  if [ -f "$PID_FILE" ]; then
    read -r value < "$PID_FILE" || true
  fi
  printf '%s\n' "$value"
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return 0
  fi
  local owner=""
  if [ -f "$LOCK_DIR/pid" ]; then
    read -r owner < "$LOCK_DIR/pid" || true
  fi
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    printf 'Another supervisor is active (pid %s)\n' "$owner" >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
}

release_lock() {
  local owner=""
  if [ -f "$LOCK_DIR/pid" ]; then
    read -r owner < "$LOCK_DIR/pid" || true
  fi
  if [ "$owner" = "$$" ]; then
    rm -rf "$LOCK_DIR"
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$rc" -ne 0 ] && [ ! -f "$FAIL_FILE" ]; then
    local stage
    stage="$(read_state)"
    printf 'supervisor exited rc=%s stage=%s\n' "$rc" "$stage" > "$FAIL_FILE"
    write_state "failed rc=$rc stage=$stage"
    log "Supervisor failed rc=$rc at stage=$stage (checkpoint retained)"
  fi
  release_lock
  exit "$rc"
}

trainer_alive() {
  pgrep -f '[n]irnay\.train' >/dev/null 2>&1
}

wait_no_trainer() {
  local timeout="${1:-120}"
  local waited=0
  while trainer_alive; do
    if [ "$waited" -ge "$timeout" ]; then
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  local old_pid=""
  old_pid="$(read_pid)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    return 1
  fi
  rm -f "$PID_FILE"
  return 0
}

free_mem_pct() {
  memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2); print int($2); found=1} END{if(!found) print 100}'
}

free_disk_gb() {
  df -k "$ROOT" | awk 'NR==2{printf "%d", $4/1024/1024}'
}

preflight() {
  # preflight quick: static prerequisites only (uv, HF_HOME, data, import).
  # Use before a long delay sleep; the full resource + idle gating happens
  # in the post-delay preflight, when the machine is actually about to train.
  if ! command -v uv >/dev/null 2>&1; then
    fail_hard "uv not found on PATH"
  fi
  if [ ! -d "$HF_HOME" ]; then
    fail_hard "HF_HOME missing: $HF_HOME"
  fi
  if [ ! -f "$ROOT/data/banking77/train.csv" ]; then
    fail_hard "banking77 train.csv missing"
  fi
  if ! uv run python -c "import nirnay, nirnay.train, nirnay.server" >/dev/null 2>&1; then
    fail_hard "nirnay import preflight failed"
  fi
  if [ "${1:-}" = "quick" ]; then
    log "Preflight quick ok (static checks only; resources gated post-delay)"
    return 0
  fi

  local waited=0
  while :; do
    local mem disk
    mem="$(free_mem_pct)"
    disk="$(free_disk_gb)"
    if [ "$mem" -ge "$MIN_FREE_MEM_PCT" ] && [ "$disk" -ge "$MIN_FREE_DISK_GB" ]; then
      log "Preflight ok: free_mem=${mem}% free_disk=${disk}GB"
      break
    fi
    if [ "$waited" -ge 1800 ]; then
      fail_hard "preflight resource wait exceeded 30m (mem=${mem}% disk=${disk}GB)"
    fi
    log "Preflight waiting: free_mem=${mem}% (min ${MIN_FREE_MEM_PCT}) free_disk=${disk}GB (min ${MIN_FREE_DISK_GB})"
    sleep 30
    waited=$((waited + 30))
  done

  if ! wait_no_trainer 300; then
    fail_hard "a nirnay.train process remained active after 300s"
  fi
}

run_uv() {
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu uv run python "$@"
  else
    uv run python "$@"
  fi
}

py_tool() {
  run_uv - "$@" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

mode = sys.argv[1]
if mode == "completed":
    meta_path = Path(sys.argv[2])
    checkpoint = Path(sys.argv[3])
    try:
        meta = json.loads(meta_path.read_text())
        value = meta.get("completed_steps", meta.get("steps", 0))
        print(int(value))
    except Exception:
        try:
            import torch
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            meta = payload.get("meta", {})
            value = meta.get("completed_steps", meta.get("steps", 0))
            print(int(value))
        except Exception:
            print(0)
elif mode == "verify":
    checkpoint = Path(sys.argv[2])
    meta_path = Path(sys.argv[3])
    target = int(sys.argv[4])
    ckpt_every = int(sys.argv[5])
    if not checkpoint.exists() or not meta_path.exists():
        raise SystemExit("checkpoint or metadata missing")
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        raise SystemExit(f"metadata unreadable: {exc}")
    completed = int(meta.get("completed_steps", meta.get("steps", 0)))
    if completed < target:
        raise SystemExit(f"checkpoint incomplete: {completed}/{target}")
    if not meta.get("encoder_frozen"):
        raise SystemExit("encoder_frozen is false")
    history_path = checkpoint.with_name("history.json")
    if not history_path.exists():
        raise SystemExit("history.json missing")
    try:
        history = json.loads(history_path.read_text())
    except Exception as exc:
        raise SystemExit(f"history unreadable: {exc}")
    if not history:
        raise SystemExit("history empty")
    if any(not math.isfinite(float(row.get("loss", float("nan")))) for row in history):
        raise SystemExit("non-finite loss in history")
    floor = max(target - ckpt_every, 1)
    if len(history) < floor:
        raise SystemExit(f"history incomplete: {len(history)}/{floor} (target {target})")
    print(f"CHECKPOINT_OK completed={completed} history={len(history)}")
else:
    raise SystemExit(f"unknown mode: {mode}")
PY
}

completed_steps() {
  py_tool completed "$CHECKPOINT_META" "$CHECKPOINT"
}

verify_completed() {
  local target="$1"
  py_tool verify "$CHECKPOINT" "$CHECKPOINT_META" "$target" "$CHECKPOINT_EVERY"
}

kill_trainers() {
  local pids
  pids="$(pgrep -f '[n]irnay\.train' 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    log "Watchdog killing stalled trainer pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 5
    pids="$(pgrep -f '[n]irnay\.train' 2>/dev/null || true)"
    if [ -n "$pids" ]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

progress_mtime() {
  local newest=0 m
  for f in "$CHECKPOINT_META" "$HISTORY_FILE" "$CHECKPOINT"; do
    if [ -f "$f" ]; then
      m="$(stat -f %m "$f" 2>/dev/null || echo 0)"
      if [ "$m" -gt "$newest" ]; then
        newest="$m"
      fi
    fi
  done
  printf '%s\n' "$newest"
}

start_watchdog() {
  local stamp
  stamp="$(progress_mtime)"
  local started
  started="$(date +%s)"
  (
    while :; do
      sleep 60
      local now current baseline age
      now="$(date +%s)"
      current="$(progress_mtime)"
      if [ "$current" -gt "$stamp" ]; then
        stamp="$current"
        started="$now"
      fi
      if [ "$stamp" -eq 0 ]; then
        baseline="$started"
      else
        baseline="$started"
      fi
      age=$((now - baseline))
      if [ "$age" -gt "$STALL_SECONDS" ]; then
        printf '[%s] WATCHDOG stall age=%ss threshold=%ss\n' \
          "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$age" "$STALL_SECONDS" \
          >> "$LOG_DIR/phase_a_watchdog.log"
        kill_trainers
        break
      fi
      if ! trainer_alive; then
        break
      fi
    done
  ) &
  WATCHDOG_PID=$!
}

stop_watchdog() {
  if [ -n "${WATCHDOG_PID:-}" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  WATCHDOG_PID=""
}

run_trainer() {
  local args=("$@")
  start_watchdog
  local rc=0
  set +e
  run_uv "${args[@]}"
  rc=$?
  set -e
  stop_watchdog
  return "$rc"
}

phase_a_command() {
  local target="$1"
  local completed
  completed="$(completed_steps)" || completed=0
  if [ "$completed" -ge "$target" ]; then
    log "Phase A already at $completed/$target; skipping"
    return 0
  fi
  if [ -f "$CHECKPOINT" ] && [ "$completed" -eq 0 ]; then
    printf '%s\n' 'Existing checkpoint has no usable completion metadata; refusing to overwrite' >&2
    return 2
  fi
  local args=(
    -m nirnay.train --phase a --steps "$target" --batch-size "$BATCH_SIZE"
    --n-synth 512 --banking-dir data/banking77 --out-dir "$ARTIFACT_DIR"
    --lora-rank 8 --lr "$LEARNING_RATE" --seed 13 --checkpoint-every "$CHECKPOINT_EVERY"
    --log-every 50 --device "$DEVICE"
  )
  if [ -n "$PRETRAINED_LR" ]; then
    args+=(--pretrained-lr "$PRETRAINED_LR")
  fi
  if [ -n "$CONCEPTS_LR" ]; then
    args+=(--concepts-lr "$CONCEPTS_LR")
  fi
  if [ "${PROBE_EVERY:-$CHECKPOINT_EVERY}" -gt 0 ]; then
    args+=(--probe-every "${PROBE_EVERY:-$CHECKPOINT_EVERY}" --probe-size 256)
  fi
  if [ -n "$N_SYNTH_HARD" ]; then
    args+=(--n-synth-hard "$N_SYNTH_HARD")
  fi
  if [ -f "$CHECKPOINT" ]; then
    args+=(--resume)
  fi
  log "Starting Phase A target=$target resume=$([ -f "$CHECKPOINT" ] && printf yes || printf no)"
  run_trainer "${args[@]}"
}

backoff_sleep() {
  local attempt="$1"
  local delay=30
  local i=1
  while [ "$i" -lt "$attempt" ]; do
    delay=$((delay * 2))
    i=$((i + 1))
  done
  if [ "$delay" -gt 600 ]; then
    delay=600
  fi
  printf '%s\n' "$delay"
}

run_resilient() {
  local label="$1"
  local target="$2"
  local attempt=1
  while [ "$attempt" -le "$((MAX_RESTARTS + 1))" ]; do
    if ! wait_no_trainer 180; then
      log "$label attempt $attempt: trainer process still active; waiting"
      sleep 30
      attempt=$((attempt + 1))
      continue
    fi
    local rc=0
    set +e
    phase_a_command "$target"
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
      if verify_completed "$target" >/dev/null; then
        verify_completed "$target"
        return 0
      fi
      log "$label attempt $attempt: training returned ok but verification failed"
    elif [ "$rc" -eq 2 ]; then
      fail_hard "$label refused to run: existing checkpoint has no usable metadata"
    else
      log "$label attempt $attempt: training exited rc=$rc"
    fi
    if [ "$attempt" -gt "$MAX_RESTARTS" ]; then
      fail_hard "$label failed after $attempt attempts"
    fi
    local delay
    delay="$(backoff_sleep "$attempt")"
    log "$label attempt $attempt failed; checkpoint retained; retrying in ${delay}s"
    sleep "$delay"
    attempt=$((attempt + 1))
  done
  fail_hard "$label exhausted retries"
}

run_resilient_eval() {
  [ "$RUN_EVAL" -eq 1 ] || return 0
  local attempt=1
  while [ "$attempt" -le "$((MAX_RESTARTS + 1))" ]; do
    if ! wait_no_trainer 180; then
      log "evaluation attempt $attempt: trainer still active; waiting"
      sleep 30
      attempt=$((attempt + 1))
      continue
    fi
    log "Starting Banking77 checkpoint evaluation attempt=$attempt limit=$EVAL_LIMIT"
    local rc=0
    set +e
    if command -v caffeinate >/dev/null 2>&1; then
      caffeinate -dimsu uv run python scripts/eval_checkpoint.py \
        --checkpoint "$CHECKPOINT" \
        --data-dir data/banking77 \
        --batch-size "$BATCH_SIZE" \
        --limit "$EVAL_LIMIT" \
        --device "$DEVICE"
    else
      uv run python scripts/eval_checkpoint.py \
        --checkpoint "$CHECKPOINT" \
        --data-dir data/banking77 \
        --batch-size "$BATCH_SIZE" \
        --limit "$EVAL_LIMIT" \
        --device "$DEVICE"
    fi
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
      return 0
    fi
    if [ "$attempt" -gt "$MAX_RESTARTS" ]; then
      fail_hard "evaluation failed after $attempt attempts; checkpoint retained"
    fi
    local delay
    delay="$(backoff_sleep "$attempt")"
    log "evaluation failed rc=$rc; retrying in ${delay}s"
    sleep "$delay"
    attempt=$((attempt + 1))
  done
  fail_hard "evaluation exhausted retries"
}

run_phase_b() {
  [ "$RUN_PHASE_B" -eq 1 ] || return 0
  verify_completed "$TOTAL_STEPS" >/dev/null
  log "Starting Phase B from $CHECKPOINT"
  local rc=0
  set +e
  run_trainer -m nirnay.train --phase b --steps 50 --batch-size "$BATCH_SIZE" \
    --n-synth 512 --banking-dir data/banking77 --out-dir "$ARTIFACT_DIR" \
    --lora-rank 8 --lr 0.0001 --seed 13 --device "$DEVICE" --group-size 4 \
    --phase-a-path "$CHECKPOINT"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    fail_hard "Phase B failed rc=$rc"
  fi
  log "Phase B completed"
}

acquire_lock
trap on_exit EXIT INT TERM
preflight quick
write_state "waiting delay_seconds=$DELAY_SECONDS"
log "Supervisor started; waiting ${DELAY_SECONDS}s"
if [ "$DELAY_SECONDS" -gt 0 ]; then
  sleep "$DELAY_SECONDS"
fi
if ! wait_no_trainer 300; then
  fail_hard "trainer active after delay; refusing to start"
fi
preflight
write_state "canary"
run_resilient canary "$CANARY_STEPS"
write_state "full"
run_resilient full "$TOTAL_STEPS"
write_state "evaluation"
run_resilient_eval
run_phase_b
verify_completed "$TOTAL_STEPS"
write_state "complete"
log "Workflow complete; checkpoint retained at $CHECKPOINT"
