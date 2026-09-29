#!/usr/bin/env tsx
/** Aggregate results/bench_*.json into results/REPORT.md with the pre-registered verdict. */
import { echo } from "zx";
import { readFileSync, readdirSync, writeFileSync, existsSync } from "node:fs";

type Row = {
  model: string; benchmark: string; n: number; f1: number; precision: number;
  recall: number; accuracy: number; ms_per_sample: number; positive_rate: number;
};

const MODELS = [
  { tag: "sentinel_v2", name: "sentinel-v2 (Qwen3-0.6B)" },
  { tag: "laya_base", name: "laya base (zero-shot)" },
  { tag: "laya_tuned", name: "laya fine-tuned (Sentinel recipe)" },
  { tag: "laya_distill", name: "laya distilled from sentinel-v2" },
  { tag: "laya_distill_extra", name: "laya distilled plus gold inject labels" },
];
const BENCHES = [
  { tag: "rogue", name: "rogue-security/prompt-injections-benchmark" },
  { tag: "wildjailbreak", name: "allenai/wildjailbreak" },
  { tag: "jackhhao", name: "jackhhao/jailbreak-classification" },
  { tag: "deepset", name: "deepset/prompt-injections" },
  { tag: "xtram1", name: "xTRam1/safe-guard-prompt-injection" },
];

const rows = new Map<string, Row>();
for (const f of readdirSync("results")) {
  if (!f.startsWith("bench_") || !f.endsWith(".json")) continue;
  const r: Row = JSON.parse(readFileSync(`results/${f}`, "utf8"));
  rows.set(`${r.model}|${r.benchmark}`, r);
}

const get = (m: string, b: string) => rows.get(`${m}|${b}`);
const fmt = (x?: number) => (x === undefined ? "-" : x.toFixed(3));

function avgF1(model: string): number | undefined {
  const vals = BENCHES.map((b) => get(model, b.tag)?.f1).filter((x): x is number => x !== undefined);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : undefined;
}

let md = `# Benchmark: sentinel-v2 vs laya base, recipe fine-tune, and distillation

Metric: **Binary F1** (positive = jailbreak/prompt-injection, threshold 0.5), measured locally on 2x RTX 3090 (deezx).
All models answered identical inputs in the same environment. The five-benchmark average excludes the internal hold-out.

## Main table (Binary F1)

| Model | ${BENCHES.map((b) => b.name).join(" | ")} | **Avg** |
|---|---|---|---|---|---|---|
`;
for (const m of MODELS) {
  const cells = BENCHES.map((b) => fmt(get(m.tag, b.tag)?.f1));
  md += `| ${m.name} | ${cells.join(" | ")} | **${fmt(avgF1(m.tag))}** |\n`;
}

md += `
## Per-benchmark delta vs sentinel-v2 (F1)

| Model | ${BENCHES.map((b) => b.name).join(" | ")} |
|---|---|---|---|---|
`;
const sAvg = avgF1("sentinel_v2");
for (const m of MODELS.slice(1)) {
  const cells = BENCHES.map((b) => {
    const s = get("sentinel_v2", b.tag)?.f1, v = get(m.tag, b.tag)?.f1;
    return s === undefined || v === undefined ? "-" : (v - s >= 0 ? "+" : "") + (v - s).toFixed(3);
  });
  md += `| ${m.name} | ${cells.join(" | ")} |\n`;
}

md += `
## Secondary metrics

| Model | Benchmark | n | Pos. rate | Precision | Recall | Accuracy | ms/sample |
|---|---|---|---|---|---|---|---|
`;
for (const m of MODELS)
  for (const b of [...BENCHES, { tag: "internal", name: "internal held-out" }]) {
    const r = get(m.tag, b.tag);
    if (r) md += `| ${m.name} | ${b.name} | ${r.n} | ${r.positive_rate} | ${fmt(r.precision)} | ${fmt(r.recall)} | ${fmt(r.accuracy)} | ${r.ms_per_sample} |\n`;
  }

function verdictFor(tag: string, label: string): string {
  const avg = avgF1(tag);
  if (sAvg === undefined || avg === undefined) return `${label}: INCOMPLETE (missing results)`;
  const rel = Math.abs(avg - sAvg) / sAvg;
  if (avg > sAvg) return `${label}: SURPRISE WIN — avg F1 ${avg.toFixed(4)} beats sentinel-v2 ${sAvg.toFixed(4)}.`;
  if (rel <= 0.05) return `${label}: WIN — avg F1 ${avg.toFixed(4)} is within 5% of sentinel-v2 ${sAvg.toFixed(4)} (rel. diff ${(rel * 100).toFixed(2)}%).`;
  return `${label}: FAIL — avg F1 ${avg.toFixed(4)} trails sentinel-v2 ${sAvg.toFixed(4)} by ${(rel * 100).toFixed(2)}% (> 5% margin).`;
}

md += `
## Verdict (pre-registered criteria)

- Win: candidate avg F1 within 5% relative of sentinel-v2 on the five external benchmarks. Surprise win: strictly beats. Fail: otherwise. Internal hold-out is reported and not included in the average.

**${verdictFor("laya_tuned", "Recipe fine-tune")}**

**${verdictFor("laya_distill", "Distillation")}**

**${verdictFor("laya_distill_extra", "Distillation plus gold inject labels")}**

## Caveats

- Sentinel v2's exact training data is unpublished ("3x more data than v1" + private qualifire-synthetics). Laya was fine-tuned on the public v1 recipe from arXiv:2506.05446 (7 public sources, 70/30 benign/attack, 90/10 split).
- verazuo/jailbreak-llms was deleted from HuggingFace; the identical CSVs were sourced from the authors' GitHub (verazuo/jailbreak_llms).
- Laya answers one fixed noul question (identical wording for training and both laya evals); sentinel-v2 is a native binary classifier.
- Latency is wall-clock per sample at batch size 64 on this machine, not the vendors' numbers.
- Fine-tuning used a full-attention override of ModernBERT (sliding-window SDPA backward deadlocks on this sm_86 stack). Eval of the tuned checkpoint uses that same config.
- The tuned model fits the recipe hold-out (internal F1 0.974) but the 5-benchmark average fell vs laya base (0.692 vs 0.769), driven by a recall collapse on deepset (R=0.11).
- Distillation trains on sentinel-v2 soft labels over a disjoint pool (v1 recipe sources, Salad non-O5, toxic-chat non-jailbreak, wildjailbreak *train*). The five eval sets, wildjailbreak eval, and the recipe internal hold-out are hash-blocked and never used as distillation inputs. Eval still uses gold labels.
`;

if (existsSync("data/dataset_report.json")) {
  const d = JSON.parse(readFileSync("data/dataset_report.json", "utf8"));
  md += `\n## Training data composition\n\n| Source | Attack | Benign |\n|---|---|---|\n`;
  for (const [src, c] of Object.entries<any>(d.per_source)) md += `| ${src} | ${c.attack} | ${c.benign} |\n`;
  md += `\nTrain items: ${d.train_items} | Held-out test: ${d.test_rows} | Attack total: ${d.attack_total} | Benign total: ${d.benign_total}\n`;
}

const goldPath = "results/gold_holdout_laya_distill_extra.json";
if (existsSync(goldPath)) {
  const g = JSON.parse(readFileSync(goldPath, "utf8"));
  md += `\n## Inject gold hold-out\n\nThis slice is 5% of the new inject rows only. It was not trained on. The label is the dataset's own hard label. It is not included in the five-benchmark average.\n\nBinary F1 ${fmt(g.f1)} | precision ${fmt(g.precision)} | recall ${fmt(g.recall)} | n ${g.n}\n`;
}

if (existsSync("data/distill_report.json")) {
  const d = JSON.parse(readFileSync("data/distill_report.json", "utf8"));
  md += `\n## Distillation pool\n\nTrain items: ${d.train_items} | Teacher hold-out: ${d.holdout_items} | Blocklist drops: ${d.dropped_blocklist} | Train teacher-positive rate: ${Number(d.train_teacher_pos_rate).toFixed(3)}\n\n| Source | n | Teacher positive |\n|---|---|---|\n`;
  for (const [src, c] of Object.entries<any>(d.per_source ?? {})) md += `| ${src} | ${c.n} | ${c.teacher_pos} |\n`;
}

writeFileSync("results/REPORT.md", md);
echo(md);
echo`REPORT_OK`;
