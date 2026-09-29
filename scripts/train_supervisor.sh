#!/bin/bash
# Resilient training supervisor for deezx.
# Root cause of the historical crashes/hangs: torch.cuda.empty_cache() called
# mid-run deadlocked the SDPA backward kernel on sm_86. That call is gone, so
# training is stable on the .venv-stable stack (torch 2.5.1+cu124). This
# supervisor remains as cheap insurance for the long run: it relaunches with
# --resume on any nonzero exit and kills/restarts if the log stops growing.
cd ~/sentinel-laya
OUT=models/laya-sentinel
LOG=results/train_log.txt
SLOG=results/train_supervisor.log

for attempt in $(seq 1 60); do
  echo "[supervisor] attempt $attempt $(date -u +%H:%M:%S)" >> "$SLOG"
  if [ -f "$OUT/checkpoint_latest/resume.pt" ]; then RESUME="--resume"; else RESUME=""; fi
  MASTER_PORT=31$((100 + RANDOM % 800)) \
    .venv-stable/bin/python python/train_rlcd.py \
    --items data/train_items.pt --output "$OUT" --single-gpu \
    --epochs 4 --ckpt-every 100 $RESUME >> "$LOG" 2>&1 &
  WPID=$!

  # Watchdog: kill the worker if the log file stops growing for 180s (hang),
  # or let it exit on its own (crash). Log size (not step number) is used
  # because step numbers reset each epoch on resume.
  last=$(stat -c%s "$LOG" 2>/dev/null); last=${last:-0}; stall=0
  while kill -0 "$WPID" 2>/dev/null; do
    sleep 20
    cur=$(stat -c%s "$LOG" 2>/dev/null); cur=${cur:-0}
    if [ "$cur" -gt "$last" ]; then last=$cur; stall=0; else stall=$((stall+20)); fi
    if [ "$stall" -ge 180 ]; then
      echo "[supervisor] attempt $attempt hung (log stalled at ${cur}B), killing" >> "$SLOG"
      kill -9 "$WPID" 2>/dev/null
      break
    fi
  done
  wait "$WPID" 2>/dev/null

  if grep -q TRAIN_OK "$LOG"; then
    echo "[supervisor] TRAIN_OK on attempt $attempt" >> "$SLOG"
    exit 0
  fi
  echo "[supervisor] attempt $attempt ended, retrying" >> "$SLOG"
  sleep 5
done
echo "[supervisor] exhausted retries" >> "$SLOG"
exit 1
