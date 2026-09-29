"""Evaluate rogue-security/prompt-injection-jailbreak-sentinel-v2 on all benchmarks.

  .venv/bin/python python/eval_sentinel.py [--benchmarks deepset,jackhhao,...] [--limit N]
"""
import argparse
import os

import torch
from transformers import pipeline

from common import BENCHMARKS, Timer, save_result

MODEL_ID = "rogue-security/prompt-injection-jailbreak-sentinel-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS.keys()))
    ap.add_argument("--limit", type=int, default=0, help="cap samples per benchmark (smoke test)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pipe = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        torch_dtype=torch.float16,
        device=0,
        truncation=True,
        max_length=4096,
    )
    # sentinel-v2's tokenizer has no pad token; reuse EOS so batched pipeline
    # inference works. Set it on the live tokenizer, the model config, and the
    # generation config so the pipeline's internal collate sees it.
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    if getattr(pipe.model.config, "pad_token_id", None) is None:
        pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
    id2label = {int(k): v.lower() for k, v in pipe.model.config.id2label.items()}
    print("id2label:", id2label, flush=True)

    for bench in args.benchmarks.split(","):
        texts, labels = BENCHMARKS[bench]()
        if args.limit:
            texts, labels = texts[: args.limit], labels[: args.limit]
        with Timer() as t:
            outs = pipe(texts, batch_size=args.batch_size)
        preds = [1 if "jailbreak" in str(o["label"]).lower() else 0 for o in outs]
        save_result(args.out, "sentinel_v2", bench, texts, labels, preds, t.elapsed)

    print("EVAL_SENTINEL_OK", flush=True)


if __name__ == "__main__":
    main()
