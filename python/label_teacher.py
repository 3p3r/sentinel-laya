"""Label a prompt pool with sentinel-v2 soft P(jailbreak).

Writes data/distill_raw.parquet: text, source, p_jailbreak, teacher_label.
Resumes if --out already exists (skips texts already labeled).
"""
import argparse
import os
import time

import pandas as pd
import torch
from transformers import pipeline

MODEL_ID = "rogue-security/prompt-injection-jailbreak-sentinel-v2"


def _jailbreak_prob(scores):
    """scores: list[{label, score}] covering both classes."""
    p = None
    other = None
    for s in scores:
        label = str(s["label"]).lower()
        if "jailbreak" in label:
            p = float(s["score"])
        else:
            other = float(s["score"])
    if p is None and other is not None:
        p = 1.0 - other
    if p is None:
        raise RuntimeError(f"no class score in {scores}")
    return min(1.0, max(0.0, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/distill_pool.parquet")
    ap.add_argument("--out", default="data/distill_raw.parquet")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="label only the first N unlabeled rows")
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool)
    done = set()
    prev = None
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done = set(prev["text"].tolist())
        print(f"resume: {len(done)} already labeled", flush=True)

    todo = pool[~pool["text"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"labeling {len(todo)} / pool {len(pool)}", flush=True)

    if len(todo) == 0:
        print("LABEL_OK", flush=True)
        return

    pipe = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        torch_dtype=torch.float16,
        device=0,
        truncation=True,
        max_length=4096,
    )
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    if getattr(pipe.model.config, "pad_token_id", None) is None:
        pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
    print("id2label:", {int(k): v for k, v in pipe.model.config.id2label.items()}, flush=True)

    texts = todo["text"].tolist()
    sources = todo["source"].tolist()
    probs = []
    bs = args.batch_size
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def _flush(upto):
        nonlocal prev
        if upto <= 0:
            return
        fresh = pd.DataFrame({
            "text": texts[:upto],
            "source": sources[:upto],
            "p_jailbreak": probs[:upto],
            "teacher_label": [1 if p >= 0.5 else 0 for p in probs[:upto]],
        })
        # texts[:upto] includes rows already flushed; rebuild from prev + all so far
        # only on the first process. Simpler: write prev (rows from earlier runs)
        # plus everything labeled in this process.
        out_df = pd.concat([prev, fresh], ignore_index=True) if prev is not None else fresh
        tmp = args.out + ".tmp"
        out_df.to_parquet(tmp)
        os.replace(tmp, args.out)

    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        outs = pipe(chunk, batch_size=bs, top_k=2)
        for o in outs:
            scores = o if isinstance(o, list) else [o]
            probs.append(_jailbreak_prob(scores))
        done_n = min(i + bs, len(texts))
        if (i // bs) % 50 == 0:
            _flush(done_n)
            print(
                f"  labeled {done_n}/{len(texts)} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    _flush(len(texts))
    out = pd.read_parquet(args.out)
    pos = int(out["teacher_label"].sum())
    print(
        f"wrote {len(out)} rows pos={pos} ({pos / max(1, len(out)):.3f}) "
        f"in {elapsed:.0f}s -> {args.out}",
        flush=True,
    )
    print("LABEL_OK", flush=True)


if __name__ == "__main__":
    main()
