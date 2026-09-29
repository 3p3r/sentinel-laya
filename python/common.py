"""Shared benchmark loaders + metrics for the 3-way prompt-injection eval.

Label convention everywhere: 1 = jailbreak/prompt-injection, 0 = benign.
"""
import json
import time

import pandas as pd
from datasets import load_dataset


def _to01(v):
    """Normalize a label that may be int, numeric string, or text like 'jailbreak'."""
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "jailbreak", "injection", "malicious", "attack", "harmful"):
        return 1
    return 0


def _deepset():
    ds = load_dataset("deepset/prompt-injections")
    rows = list(ds["train"]) + list(ds["test"])
    return [r["text"] for r in rows], [_to01(r["label"]) for r in rows]


def _jackhhao():
    ds = load_dataset("jackhhao/jailbreak-classification")
    rows = list(ds["train"]) + list(ds["test"])
    return [r["prompt"] for r in rows], [1 if str(r["type"]).strip().lower() == "jailbreak" else 0 for r in rows]


def _xtram1():
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    rows = list(ds["train"]) + list(ds["test"])
    return [r["text"] for r in rows], [_to01(r["label"]) for r in rows]


def _rogue():
    ds = load_dataset("rogue-security/prompt-injections-benchmark", split="test")
    return [r["text"] for r in ds], [_to01(r["label"]) for r in ds]


def _wildjailbreak():
    # the "eval" config of allenai/wildjailbreak exposes a single split named "train"
    ds = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False, split="train")
    texts, labels = [], []
    for r in ds:
        t = r.get("adversarial") or r.get("vanilla") or r.get("prompt") or ""
        if not t:
            continue
        texts.append(t)
        labels.append(1 if "harmful" in str(r["data_type"]) else 0)
    return texts, labels


def _internal():
    df = pd.read_parquet("data/internal_test.parquet")
    return df["text"].tolist(), [int(x) for x in df["label"]]


BENCHMARKS = {
    "deepset": _deepset,
    "jackhhao": _jackhhao,
    "xtram1": _xtram1,
    "rogue": _rogue,
    "wildjailbreak": _wildjailbreak,
    "internal": _internal,
}


def metrics(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "accuracy": round(acc, 4)}


def save_result(out_dir, model_tag, bench, texts, labels, preds, elapsed_s):
    m = metrics(preds, labels)
    m.update({
        "model": model_tag,
        "benchmark": bench,
        "n": len(labels),
        "positive_rate": round(sum(labels) / max(1, len(labels)), 4),
        "elapsed_s": round(elapsed_s, 2),
        "ms_per_sample": round(1000 * elapsed_s / max(1, len(labels)), 2),
    })
    with open(f"{out_dir}/bench_{model_tag}_{bench}.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"  {model_tag} / {bench}: F1={m['f1']} acc={m['accuracy']} n={m['n']} ({m['ms_per_sample']} ms/sample)", flush=True)
    return m


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
