"""Score the inject-only gold holdout with the source label.

This number is not part of the five-benchmark average.
"""
import argparse
import json
import os

import laya
import pandas as pd

from common import Timer, metrics
from eval_laya import QUESTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", default="data/extra_gold_holdout.parquet")
    ap.add_argument("--tag", default="laya_distill_extra")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frame = pd.read_parquet(args.rows)
    texts = frame["text"].astype(str).tolist()
    labels = [int(x) for x in frame["teacher_label"].tolist()]
    agent = laya.load(args.model, device="cuda")
    agent.cfg["max_len"] = 1024
    agent.cfg["head_max_len"] = 256
    states = [{"prompt": t} for t in texts]
    with Timer() as timer:
        results = agent.predict_batch(states, QUESTIONS, batch_size=args.batch_size, sort_by_length=True)
    preds = [1 if r["answers"]["injection"]["noul"] >= 0.5 else 0 for r in results]
    row = metrics(preds, labels)
    row.update({
        "model": args.tag,
        "benchmark": "inject_gold_holdout",
        "n": len(labels),
        "positive_rate": round(sum(labels) / max(1, len(labels)), 4),
        "elapsed_s": round(timer.elapsed, 2),
        "ms_per_sample": round(1000 * timer.elapsed / max(1, len(labels)), 2),
    })
    path = os.path.join(args.out, f"gold_holdout_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2)
    print(
        f"GOLD_HOLDOUT_OK f1={row['f1']} precision={row['precision']} recall={row['recall']} n={row['n']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
