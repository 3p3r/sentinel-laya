#!/usr/bin/env tsx
/**
 * Fine-tune laya on deezx (2x 3090, DDP via torchrun).
 *   npx tsx scripts/train.ts --smoke   # 200 items, 1 epoch, foreground (~5 min)
 *   npx tsx scripts/train.ts           # full run, nohup background + polling (hours)
 */
import { echo } from "zx";
import { ssh, sshOut, push, pull, REMOTE_DIR } from "./lib.ts";

const smoke = process.argv.includes("--smoke");

echo`== push code ==`;
await push();

// NOTE: NCCL allreduce stalls on this host (2x3090, driver 580.178.04, torch 2.14+cu130)
// even with P2P/IB disabled and checkpointing off — both ranks enqueue the same
// collective but it never completes. Single-GPU mode is the reliable fallback.
// --no-ckpt: gradient checkpointing's recompute mismatches metadata on this stack;
// at 512 max_len memory fits comfortably on the 24GB 3090 without it.
const TRAIN_CMD = (out: string, extra: string) =>
  `.venv/bin/python python/train_rlcd.py --items data/train_items.pt --output ${out} --single-gpu --no-ckpt ${extra}`;

if (smoke) {
  echo`== smoke run: 200 items, 1 epoch, foreground (single-gpu) ==`;
  await ssh(`cd ~/${REMOTE_DIR} && mkdir -p results models && ${TRAIN_CMD("models/laya-sentinel-smoke", "--max-items 200 --epochs 1")} 2>&1 | tail -30`);
  echo`SMOKE_OK`;
  process.exit(0);
}

// The ModernBERT sliding-window SDPA backward is fixed by forcing full attention
// (see train_rlcd.py), but as a safety net the supervisor still auto-resumes from
// the rolling checkpoint on any crash and kills/restarts on hang.
echo`== launch resilient full training on deezx (nohup supervisor) ==`;
await sshOut(
  `cd ~/${REMOTE_DIR} && mkdir -p results models && rm -f results/train_log.txt results/train_supervisor.log && chmod +x scripts/train_supervisor.sh && nohup scripts/train_supervisor.sh > /dev/null 2>&1 & echo supervisor pid $!`
);

echo`== polling results/train_log.txt every 60s ==`;
const t0 = Date.now();
for (;;) {
  await new Promise((r) => setTimeout(r, 60_000));
  const tail = await sshOut(`tail -8 ~/${REMOTE_DIR}/results/train_log.txt 2>/dev/null || echo '(no log yet)'`);
  const alive = await sshOut(`pgrep -f train_rlcd.py >/dev/null && echo yes || echo no`);
  const supervisor = await sshOut(`pgrep -f train_supervisor.sh >/dev/null && echo yes || echo no`);
  echo`--- ${((Date.now() - t0) / 60000).toFixed(0)}min | worker alive: ${alive} | supervisor: ${supervisor} ---`;
  echo(tail);
  if (tail.includes("TRAIN_OK")) {
    echo`== training finished, pulling artifacts ==`;
    await pull("models", "models");
    await pull("results", "results");
    echo`TRAIN_DONE`;
    break;
  }
  // supervisor restarts the worker on crash; only give up if the supervisor itself is gone
  if (supervisor === "no" && !tail.includes("TRAIN_OK")) {
    const slog = await sshOut(`tail -5 ~/${REMOTE_DIR}/results/train_supervisor.log 2>/dev/null || echo '(no supervisor log)'`);
    echo`!! supervisor exited without TRAIN_OK; supervisor log:`;
    echo(slog);
    await pull("results", "results");
    process.exit(1);
  }
}
