"""Replace Sentinel scores on the inject pool with each dataset's own label.

The 220k distill file is not touched. Attack-only corpora are 1. NotInject is
benign, so 0. Harelix uses the 0/1 column in ahsanayub/malicious-prompts.
HackAPrompt `correct` and Rubend's Jailbreak Score are attack-success or
quality scores, not the jailbreak-attempt class.

Writes data/inject_raw.parquet and data/inject_gold_report.json.
Run from the repo root: python python/apply_inject_gold.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys

import pandas as pd
from datasets import load_dataset

from build_dataset import clean

POOL = "data/inject_pool.parquet"
OUT = "data/inject_raw.parquet"
REPORT = "data/inject_gold_report.json"
MIRROR = "ahsanayub/malicious-prompts"

# Class of the text for the jailbreak question, not whether an attack succeeded.
CONSTANT = {
    "vigil_instruction_bypass": 1,
    "vigil_jailbreak": 1,
    "notinject": 0,
    "rubend18": 1,
    "injecagent_instruction": 1,
    "injecagent_enhanced": 1,
    "bipia_text_attack": 1,
    "bipia_code_attack": 1,
    "hackaprompt": 1,
}

RULE = {
    "harelix": "ahsanayub/malicious-prompts label column, rows whose source contains harelix",
    "notinject": "benign trigger-word prompts; the dataset has no attack class",
    "vigil_instruction_bypass": "attack-only corpus; no label column",
    "vigil_jailbreak": "attack-only corpus; no label column",
    "rubend18": "jailbreak-prompt corpus; Jailbreak Score and GPT-4 are not the class",
    "injecagent_instruction": "attacker instruction",
    "injecagent_enhanced": "attacker instruction plus the ignore-previous prefix",
    "bipia_text_attack": "BIPIA text attack string",
    "bipia_code_attack": "BIPIA code attack string",
    "hackaprompt": "submission_data jailbreak attempt; the correct column is attack success, not the class",
}


def harelix_labels():
    ds = load_dataset(MIRROR)
    labels = {}
    conflicts = 0
    for split in ds.values():
        for src, text, label in zip(split["source"], split["text"], split["label"]):
            if src is None or "harelix" not in str(src).lower():
                continue
            t = clean(text)
            if t is None:
                continue
            lab = int(label)
            if lab not in (0, 1):
                raise SystemExit(f"harelix label {label!r} is not 0/1")
            prev = labels.get(t)
            if prev is None:
                labels[t] = lab
            elif prev != lab:
                conflicts += 1
    if conflicts:
        raise SystemExit(f"harelix cleaned text has {conflicts} conflicting labels")
    return labels


def main():
    if not os.path.isfile(POOL):
        raise SystemExit(f"missing {POOL}")
    pool = pd.read_parquet(POOL)
    if list(pool.columns) != ["text", "source"]:
        raise SystemExit(f"unexpected pool columns {list(pool.columns)}")

    hare = harelix_labels()
    gold = []
    missing = 0
    for text, source in zip(pool["text"], pool["source"]):
        source = str(source)
        if source == "harelix":
            lab = hare.get(text)
            if lab is None:
                missing += 1
                continue
        elif source in CONSTANT:
            lab = CONSTANT[source]
        else:
            raise SystemExit(f"no gold rule for source {source}")
        gold.append(lab)
    if missing:
        raise SystemExit(f"{missing} harelix rows had no label in {MIRROR}")
    if len(gold) != len(pool):
        raise SystemExit(f"labeled {len(gold)} of {len(pool)}")

    out = pd.DataFrame({
        "text": pool["text"],
        "source": pool["source"],
        "p_jailbreak": [float(v) for v in gold],
        "teacher_label": gold,
    })
    tmp = OUT + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, OUT)

    per = {}
    for source, g in out.groupby("source", sort=True):
        per[str(source)] = {
            "n": int(len(g)),
            "positives": int(g["teacher_label"].sum()),
            "rule": RULE[str(source)],
        }
    report = {
        "rows": int(len(out)),
        "label": "dataset gold, hard 0/1; p_jailbreak equals teacher_label",
        "sentinel_scores": "discarded",
        "per_source": per,
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"rows={len(out)} pos={int(out.teacher_label.sum())} -> {OUT}", flush=True)
    print("GOLD_OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
