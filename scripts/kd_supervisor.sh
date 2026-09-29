#!/bin/bash
# Retired. The distill job uses math attention and skips a bad update inside
# the same process. Starting this script would kill that job and reload.
echo "[kd-supervisor] retired; do not start a watchdog" >&2
exit 1
cd ~/sentinel-laya
mkdir -p results models/laya-distill
LOG=${KD_LOG:-results/distill_train.log}
SLOG=${KD_SLOG:-results/kd_supervisor.log}
OUTPUT=${KD_OUTPUT:-models/laya-distill}
STATE_DIR=${KD_STATE_DIR:-results}
MAX_ITEMS=${KD_MAX_ITEMS:-0}
SAVE_STEPS=${KD_SAVE_STEPS:-100}
EPOCHS=${KD_EPOCHS:-2}
export CUDA_VISIBLE_DEVICES=0
export KD_FULL_ATTN=1
export KD_MATH_SDP=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

for attempt in $(seq 1 500); do
  echo "[kd-supervisor] attempt $attempt $(date -u +%H:%M:%S)" >> "$SLOG"
  echo "KD_ATTEMPT_$attempt" >> "$LOG"
  .venv-stable/bin/python -u python/train_kd.py \
    --epochs "$EPOCHS" --output "$OUTPUT" --state-dir "$STATE_DIR" \
    --max-items "$MAX_ITEMS" --save-steps "$SAVE_STEPS" \
    --resume --precision fp32 \
    >> "$LOG" 2>&1 &
  WPID=$!
  # A wedged CUDA kernel holds the GIL, so an in-process watchdog cannot run.
  # Checkpoint writes and the 11k holdout are heartbeated; 600s without a log
  # write means the step is stuck.
  last=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  stall=0
  while kill -0 "$WPID" 2>/dev/null; do
    sleep 20
    cur=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    if [ "$cur" != "$last" ]; then last=$cur; stall=0; else stall=$((stall + 20)); fi
    if [ "$stall" -ge 600 ]; then
      echo "[kd-supervisor] attempt $attempt hung, killing $WPID" >> "$SLOG"
      kill -9 "$WPID" 2>/dev/null
      break
    fi
  done
  wait "$WPID" 2>/dev/null
  code=$?
  segment=$(awk -v token="KD_ATTEMPT_$attempt" '$0 == token { found=1; buf=""; next } found { buf = buf $0 "\n" } END { printf "%s", buf }' "$LOG")
  if printf '%s' "$segment" | grep -qE "FATAL_HOLDOUT|FATAL_RESUME|REFUSE_EXPORT|non-finite weights|non-finite optimizer|refused checkpoint"; then
    echo "[kd-supervisor] terminal numerical failure; stopped for inspection" >> "$SLOG"
    exit 2
  fi
  if printf '%s' "$segment" | grep -q TRAIN_OK; then
    echo "[kd-supervisor] TRAIN_OK on attempt $attempt" >> "$SLOG"
    exit 0
  fi
  echo "[kd-supervisor] attempt $attempt ended code=$code, retrying from last checkpoint" >> "$SLOG"
  sleep 5
done
echo "[kd-supervisor] exhausted retries" >> "$SLOG"
exit 1
