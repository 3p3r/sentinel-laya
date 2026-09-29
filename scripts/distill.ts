#!/usr/bin/env tsx
/**
 * Distill sentinel-v2 into base Laya on deezx (.venv-stable).
 *
 * Each heavy stage is started under setsid and polled, so a dropped local SSH
 * session does not kill the remote job. Re-running skips stages whose log
 * already contains the done marker.
 *
 *   npx tsx scripts/distill.ts
 */
import { echo } from "zx";
import { mkdirSync, writeFileSync } from "node:fs";
import { ssh, sshOut, push, pull, REMOTE_DIR } from "./lib.ts";

const PY = ".venv-stable/bin/python";
const ROOT = `~/${REMOTE_DIR}`;

/** Bracket the first char so `ps | grep` does not match its own command line. */
function bracket(token: string) {
  return `[${token[0]}]${token.slice(1)}`;
}

async function stage(name: string, match: string, log: string, marker: string, cmd: string) {
  const remoteLog = `${ROOT}/${log}`;
  const pat = bracket(match);
  const already = await sshOut(
    `tr "\\r" "\\n" < ${remoteLog} 2>/dev/null | grep -a ${marker} >/dev/null && echo yes || echo no`
  );
  if (already === "yes") {
    echo`== ${name}: already ${marker}, skipping ==`;
    return;
  }
  const running = await sshOut(
    `ps -eo args | grep -E '${pat}' >/dev/null && echo yes || echo no`
  );
  if (running !== "yes") {
    echo`== ${name}: launching ==`;
    await ssh(
      `cd ${ROOT} && mkdir -p results && setsid bash -lc ${JSON.stringify(cmd + ` > ${log} 2>&1`)} </dev/null >/dev/null 2>&1 & echo launched`
    );
  } else {
    echo`== ${name}: already running ==`;
  }
  const t0 = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 60_000));
    const tail = await sshOut(`tr "\\r" "\\n" < ${remoteLog} 2>/dev/null | tail -6 || echo '(no log yet)'`);
    const hit = await sshOut(
      `tr "\\r" "\\n" < ${remoteLog} 2>/dev/null | grep -a ${marker} >/dev/null && echo yes || echo no`
    );
    echo`--- ${name} ${((Date.now() - t0) / 60000).toFixed(0)}min ---`;
    echo(tail);
    if (hit === "yes") return;
    const alive = await sshOut(`ps -eo args | grep -E '${pat}' >/dev/null && echo yes || echo no`);
    if (alive === "no" && hit !== "yes") {
      const err = await sshOut(`tr "\\r" "\\n" < ${remoteLog} 2>/dev/null | tail -40`);
      echo`!! ${name} exited without ${marker}`;
      echo(err);
      process.exit(1);
    }
  }
}

echo`== push code ==`;
await push();

await stage(
  "build_pool",
  "python/build_pool.py",
  "results/distill_pool.log",
  "POOL_OK",
  `${PY} python/build_pool.py`
);

await stage(
  "label_teacher",
  "python/label_teacher.py",
  "results/distill_label.log",
  "LABEL_OK",
  `CUDA_VISIBLE_DEVICES=0 ${PY} python/label_teacher.py`
);

await stage(
  "build_distill",
  "python/build_distill.py",
  "results/distill_build.log",
  "DISTILL_DATA_OK",
  `${PY} python/build_distill.py`
);

await stage(
  "train_kd_smoke",
  "laya-distill-smoke",
  "results/distill_smoke.log",
  "TRAIN_OK",
  `CUDA_VISIBLE_DEVICES=0 KD_FULL_ATTN=1 ${PY} python/train_kd.py --max-items 3000 --epochs 1 --output /tmp/laya-distill-smoke`
);

await stage(
  "train_kd_full",
  "models/laya-distill",
  "results/distill_train.log",
  "TRAIN_OK",
  `CUDA_VISIBLE_DEVICES=0 KD_FULL_ATTN=1 ${PY} python/train_kd.py --epochs 2 --output models/laya-distill`
);

await stage(
  "eval_laya_distill",
  "tag laya_distill",
  "results/eval_laya_distill.log",
  "EVAL_LAYA_OK",
  `CUDA_VISIBLE_DEVICES=0 ${PY} python/eval_laya.py --model models/laya-distill --tag laya_distill`
);

echo`== pull results and distill artifacts ==`;
await pull("results", "results");
await pull("models/laya-distill", "models/laya-distill");
const report = await ssh(`cd ${ROOT} && cat data/distill_report.json`);
mkdirSync("data", { recursive: true });
writeFileSync("data/distill_report.json", report.stdout);
echo`DISTILL_OK`;
