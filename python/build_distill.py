"""Tokenize teacher-labeled prompts into Laya training items.

Re-applies the eval blocklist, holds out 5% for early stopping (teacher agreement
only — never gold eval), then upsamples the minority teacher-argmax class on the
train split. Soft targets [1-p, p] are kept through balancing.

Writes, unless --items-out / --holdout-out / --report-out say otherwise:
  data/distill_items.pt
  data/distill_holdout.pt
  data/distill_report.json

With --inject, the 5% holdout is taken from that file only. The --raw rows
all go into training. The inject holdout text is also written when
--holdout-text-out is set.
"""
import argparse
import json
import os
import random

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from blocklist import load_blocklist, norm_key
from build_dataset import QUESTION
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence

SEED = 42
MODEL_ID = "convaiinnovations/laya"


def _tokenize(frame, tok, max_len, head_max_len):
    items, skipped = [], 0
    for i, row in enumerate(frame.itertuples(index=False)):
        seq, markers = build_sequence(tok, {"prompt": row.text}, QUESTION, max_len, head_max_len)
        if len(markers) != 2:
            skipped += 1
            continue
        p = float(row.p_jailbreak)
        items.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES["noul"],
            "target": [1.0 - p, p],
            "label": int(row.teacher_label),
            "source": row.source,
        })
        if (i + 1) % 20000 == 0:
            print(f"  tokenized {i + 1}/{len(frame)}", flush=True)
    return items, skipped


def _upsample(frame, max_repeat=4):
    pos = frame[frame["teacher_label"] == 1]
    neg = frame[frame["teacher_label"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return frame
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    need = len(majority) - len(minority)
    cap = len(minority) * (max_repeat - 1)
    need = min(need, cap)
    extra = minority.sample(n=need, replace=True, random_state=SEED) if need else minority.iloc[0:0]
    out = pd.concat([majority, minority, extra], ignore_index=True)
    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def _filter_blocklist(raw, block):
    keep = []
    dropped = 0
    for row in raw.itertuples(index=False):
        k = norm_key(row.text)
        if k is None or k in block:
            dropped += 1
            continue
        keep.append(row)
    return pd.DataFrame(keep), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--raw", default="data/distill_raw.parquet")
    ap.add_argument("--inject", default="", help="gold-label rows; holdout is taken from this file only")
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--items-out", default="data/distill_items.pt")
    ap.add_argument("--holdout-out", default="data/distill_holdout.pt")
    ap.add_argument("--report-out", default="data/distill_report.json")
    ap.add_argument("--holdout-text-out", default="")
    args = ap.parse_args()

    raw = pd.read_parquet(args.raw)
    print(f"raw rows: {len(raw)}", flush=True)
    print("re-checking eval blocklist...", flush=True)
    block = load_blocklist()
    frame, dropped = _filter_blocklist(raw, block)
    print(f"after blocklist: {len(frame)} dropped={dropped}", flush=True)
    holdout_scope = "union"

    if args.inject:
        inject_raw = pd.read_parquet(args.inject)
        print(f"inject rows: {len(inject_raw)}", flush=True)
        inject, inject_dropped = _filter_blocklist(inject_raw, block)
        dropped += inject_dropped
        print(f"inject after blocklist: {len(inject)} dropped={inject_dropped}", flush=True)
        if len(inject) < 2:
            raise SystemExit("inject file has no rows left after the blocklist")
        inject = inject.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n_hold = max(1, int(round(len(inject) * args.holdout_frac)))
        hold_df = inject.iloc[:n_hold].reset_index(drop=True)
        train_inject = inject.iloc[n_hold:].reset_index(drop=True)
        train_df = _upsample(pd.concat([frame, train_inject], ignore_index=True))
        frame = pd.concat([frame, inject], ignore_index=True)
        holdout_scope = "inject"
    else:
        frame = frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n_hold = max(1, int(round(len(frame) * args.holdout_frac)))
        hold_df = frame.iloc[:n_hold].reset_index(drop=True)
        train_df = _upsample(frame.iloc[n_hold:].reset_index(drop=True))
    print(
        f"train={len(train_df)} (pos {(train_df.teacher_label == 1).mean():.3f}) "
        f"holdout={len(hold_df)} (pos {(hold_df.teacher_label == 1).mean():.3f}) "
        f"scope={holdout_scope}",
        flush=True,
    )

    model_dir = snapshot_download(args.model_id)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    max_len, head_max_len = 512, 256

    print("tokenizing train...", flush=True)
    train_items, skip_tr = _tokenize(train_df, tok, max_len, head_max_len)
    print("tokenizing holdout...", flush=True)
    hold_items, skip_ho = _tokenize(hold_df, tok, max_len, head_max_len)

    os.makedirs("data", exist_ok=True)
    for path in (args.items_out, args.holdout_out, args.report_out, args.holdout_text_out):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(train_items, args.items_out)
    torch.save(hold_items, args.holdout_out)
    if args.holdout_text_out:
        hold_df[["text", "source", "teacher_label"]].to_parquet(args.holdout_text_out, index=False)

    per_source = {}
    for src, g in frame.groupby("source"):
        per_source[src] = {
            "n": int(len(g)),
            "teacher_pos": int((g["teacher_label"] == 1).sum()),
        }
    report = {
        "raw_rows": int(len(raw)),
        "dropped_blocklist": dropped,
        "train_items": len(train_items),
        "holdout_items": len(hold_items),
        "holdout_scope": holdout_scope,
        "skipped_tokenize": skip_tr + skip_ho,
        "train_teacher_pos_rate": float((train_df.teacher_label == 1).mean()) if len(train_df) else 0.0,
        "per_source": per_source,
        "question": QUESTION,
    }
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"items train={len(train_items)} holdout={len(hold_items)} skipped={skip_tr + skip_ho}", flush=True)
    print("DISTILL_DATA_OK", flush=True)


if __name__ == "__main__":
    main()
