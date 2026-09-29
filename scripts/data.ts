#!/usr/bin/env tsx
/** Build the training dataset on deezx, pull the report back.
 *  npx tsx scripts/data.ts --smoke   # 200 rows/source, fast end-to-end validation
 */
import { echo } from "zx";
import { ssh, push, pull, REMOTE_DIR } from "./lib.ts";

const smoke = process.argv.includes("--smoke");

echo`== push code to deezx ==`;
await push();

echo`== build dataset on deezx (downloads several GB from HF/GitHub) ==`;
await ssh(`cd ~/${REMOTE_DIR} && .venv/bin/python python/build_dataset.py ${smoke ? "--max-per-source 200" : ""}`);

echo`== pull dataset report + internal test split back ==`;
await pull("data", "data");

echo`DATA_OK`;
