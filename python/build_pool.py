"""Unlabeled distillation pool. Gold labels are ignored; the teacher overwrites them.

Sources (none of the five eval datasets, and not the wildjailbreak *eval* config):
  - Sentinel v1 public recipe pools (build_dataset.load_sources), higher benign cap
  - Salad-Data rows outside O5 (short harmful / jailbreak-adjacent prompts)
  - lmsys/toxic-chat non-jailbreak user inputs (extra benign-ish chat)
  - allenai/wildjailbreak *train* config only

Writes data/distill_pool.parquet (text, source) after exact-text dedupe and the
eval blocklist. Caps the pool at --max-pool (default 220_000).
"""
import argparse
import random

import pandas as pd
from datasets import load_dataset

from blocklist import load_blocklist, norm_key
from build_dataset import CAP, clean, load_sources

SEED = 42


def _add(rows, text, source, seen, block):
    t = clean(text)
    if not t:
        return False
    k = t.lower()
    if k in seen or k in block:
        return False
    seen.add(k)
    rows.append((t, source))
    return True


def _salad_extra(rows, seen, block, cap):
    print("[extra] Salad-Data non-O5", flush=True)
    salad = load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train")
    picked = []
    for row in salad:
        if str(row["1-category"]).startswith("O5"):
            continue
        t = clean(row["question"])
        if t:
            picked.append(t)
    random.Random(SEED).shuffle(picked)
    n = 0
    for t in picked:
        if n >= cap:
            break
        if _add(rows, t, "salad_other", seen, block):
            n += 1
    print(f"  salad_other: {n}", flush=True)


def _toxic_benign(rows, seen, block, cap):
    print("[extra] toxic-chat non-jailbreak", flush=True)
    n = 0
    for split in ("train", "test"):
        tox = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
        for row in tox:
            if n >= cap:
                break
            if int(row["jailbreaking"]) == 1:
                continue
            if _add(rows, row["user_input"], "toxic_chat_other", seen, block):
                n += 1
    print(f"  toxic_chat_other: {n}", flush=True)


def _wildjailbreak_train(rows, seen, block, cap_each):
    print("[extra] allenai/wildjailbreak train config (not eval)", flush=True)
    ds = load_dataset(
        "allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False, split="train"
    )
    adv, van = [], []
    for row in ds:
        a = row.get("adversarial") or ""
        v = row.get("vanilla") or ""
        if isinstance(a, str) and a.strip():
            adv.append(a)
        if isinstance(v, str) and v.strip() and v.strip() != (a or "").strip():
            van.append(v)
    random.Random(SEED).shuffle(adv)
    random.Random(SEED + 1).shuffle(van)
    na = nv = 0
    for t in adv:
        if na >= cap_each:
            break
        if _add(rows, t, "wildjailbreak_train_adv", seen, block):
            na += 1
    for t in van:
        if nv >= cap_each:
            break
        if _add(rows, t, "wildjailbreak_train_van", seen, block):
            nv += 1
    print(f"  wildjailbreak_train adv={na} vanilla={nv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pool", type=int, default=220_000)
    ap.add_argument("--benign-cap", type=int, default=40_000)
    ap.add_argument("--out", default="data/distill_pool.parquet")
    args = ap.parse_args()

    # CAP is the smoke cap inside load_sources; 0 means "no per-source attack cap".
    assert CAP == 0 or CAP > 0

    print("loading eval blocklist...", flush=True)
    block = load_blocklist()

    print("loading v1 recipe pools (labels discarded)...", flush=True)
    attack, benign, verazuo = load_sources(benign_cap=args.benign_cap)
    rows, seen = [], set()
    for text, src in attack + benign + verazuo:
        _add(rows, text, src, seen, block)
    print(f"  recipe kept: {len(rows)}", flush=True)

    _salad_extra(rows, seen, block, cap=15_000)
    _toxic_benign(rows, seen, block, cap=8_000)
    _wildjailbreak_train(rows, seen, block, cap_each=50_000)

    if len(rows) > args.max_pool:
        random.Random(SEED).shuffle(rows)
        rows = rows[: args.max_pool]
    print(f"pool={len(rows)} -> {args.out}", flush=True)

    pd.DataFrame(rows, columns=["text", "source"]).to_parquet(args.out)
    print("POOL_OK", flush=True)


if __name__ == "__main__":
    main()
