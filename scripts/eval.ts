#!/usr/bin/env tsx
/**
 * Run the 3-model x 5-benchmark eval on deezx and pull results back.
 *   npx tsx scripts/eval.ts --smoke   # 50 rows/benchmark, sentinel + base laya only
 *   npx tsx scripts/eval.ts           # full: sentinel-v2, laya base, laya fine-tuned
 */
import { echo } from "zx";
import { ssh, push, pull, REMOTE_DIR } from "./lib.ts";

const smoke = process.argv.includes("--smoke");
const limit = smoke ? "--limit 50" : "";
const benches = smoke ? "deepset,jackhhao,internal" : "deepset,jackhhao,xtram1,rogue,wildjailbreak,internal";

echo`== push code ==`;
await push();

echo`== 1/3 sentinel-v2 ==`;
await ssh(`cd ~/${REMOTE_DIR} && .venv/bin/python python/eval_sentinel.py --benchmarks ${benches} ${limit}`);

echo`== 2/3 laya base ==`;
await ssh(`cd ~/${REMOTE_DIR} && .venv/bin/python python/eval_laya.py --model convaiinnovations/laya --tag laya_base --benchmarks ${benches} ${limit}`);

if (!smoke) {
  echo`== 3/3 laya fine-tuned ==`;
  await ssh(`cd ~/${REMOTE_DIR} && .venv/bin/python python/eval_laya.py --model models/laya-sentinel --tag laya_tuned --benchmarks ${benches} ${limit}`);
}

echo`== pull results ==`;
await pull("results", "results");

echo`EVAL_OK`;
