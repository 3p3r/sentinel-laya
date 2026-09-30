"""Build buried-attack rows from the existing distill pool.

A benign question is split near the middle and a short attack is inserted.
The new row is an attack. Long jailbreaks are not used. Eval texts are skipped.
"""
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from blocklist import load_blocklist, norm_key

SEED = 42
N = 4000
RAW = "data/distill_raw.parquet"
OUT = "data/bury_raw.parquet"


def _split_middle(text):
    mid = len(text) // 2
    left_at = text.rfind(" ", 0, mid)
    right_at = text.find(" ", mid)
    if left_at < 1 and right_at < 0:
        return None
    if left_at < 1:
        cut = right_at
    elif right_at < 0:
        cut = left_at
    elif (mid - left_at) <= (right_at - mid):
        cut = left_at
    else:
        cut = right_at
    left, right = text[:cut].strip(), text[cut:].strip()
    if not left or not right:
        return None
    return left, right


def main():
    frame = pd.read_parquet(RAW)
    print(f"raw rows: {len(frame)}", flush=True)
    print("loading eval blocklist...", flush=True)
    block = load_blocklist()
    benign, attacks = [], []
    for row in frame.itertuples(index=False):
        text = str(row.text)
        key = norm_key(text)
        if not key or key in block:
            continue
        n = len(text)
        if int(row.teacher_label) == 0 and 80 <= n <= 400:
            benign.append(text)
        elif int(row.teacher_label) == 1 and 20 <= n <= 160:
            attacks.append(text)
    print(f"carriers={len(benign)} short_attacks={len(attacks)}", flush=True)
    if len(benign) < 2 or len(attacks) < 2:
        raise SystemExit("not enough short rows to build buried attacks")

    rng = random.Random(SEED)
    seen = set()
    rows = []
    tries = 0
    while len(rows) < N and tries < N * 40:
        tries += 1
        parts = _split_middle(rng.choice(benign))
        if parts is None:
            continue
        attack = rng.choice(attacks)
        text = f"{parts[0]} {attack} {parts[1]}"
        key = norm_key(text)
        if not key or key in block or key in seen:
            continue
        seen.add(key)
        rows.append({
            "text": text,
            "source": "bury",
            "p_jailbreak": 1.0,
            "teacher_label": 1,
        })
    if len(rows) < N:
        raise SystemExit(f"only built {len(rows)} buried rows")
    out = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"rows={len(out)} -> {OUT}", flush=True)
    print("BURY_OK", flush=True)


if __name__ == "__main__":
    main()
