"""Eval-set text blocklist. Distillation must never train on these strings.

Same normalization as build_dataset.clean(), then lowercase. Covers every
benchmark loader in common.BENCHMARKS (the five external sets, both splits
where we eval both, wildjailbreak eval, and the recipe internal hold-out).
"""
from build_dataset import clean
from common import BENCHMARKS


def norm_key(text):
    t = clean(text)
    return None if t is None else t.lower()


def load_blocklist():
    keys = set()
    for name, fn in BENCHMARKS.items():
        texts, _labels = fn()
        n = 0
        for text in texts:
            k = norm_key(text)
            if k:
                keys.add(k)
                n += 1
        print(f"  blocklist {name}: {n}", flush=True)
    print(f"blocklist size: {len(keys)}", flush=True)
    return keys
