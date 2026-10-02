"""Compose short role and pivot rows from agent-written pieces.

Reads data/short_role_pieces.json. Writes data/short_role_raw.parquet and
data/short_role_report.json. Drops eval-blocklist hits and German rows that
are not German.
"""
import json
import os
import random
import re

import pandas as pd

from blocklist import load_blocklist, norm_key

SEED = 42
MAX_CHARS = 200
PIECES = "data/short_role_pieces.json"
OUT = "data/short_role_raw.parquet"
REPORT = "data/short_role_report.json"

QUOTAS = [
    ("en", "role", 1, 1000),
    ("de", "role", 1, 1000),
    ("en", "pivot", 1, 1000),
    ("de", "pivot", 1, 1000),
    ("en", "role", 0, 500),
    ("de", "role", 0, 500),
    ("en", "pivot", 0, 500),
    ("de", "pivot", 0, 500),
]

EN_LEFTOVERS = {
    "the", "how", "what", "which", "should", "with", "for", "and", "that",
    "this", "are", "timed", "keep", "when", "where", "from", "your", "you",
    "now", "forget", "ignore", "answer", "please",
}


DE_MARKERS = {
    "der", "die", "das", "und", "nicht", "du", "bist", "sie", "ist", "ein",
    "eine", "jetzt", "bitte", "wie", "was", "ich", "danke", "welche", "für",
    "mit", "auf", "den", "dem", "ein", "einem", "einer", "dass", "auch",
    "wir", "ihr", "dein", "deine", "schon", "noch", "oder", "aber", "von",
    "zu", "im", "am", "zum", "zur", "nach", "bei", "aus", "wenn", "dann",
}


def _words(text):
    return set(re.findall(r"[a-zäöüß]+", text.lower()))


def _german(text):
    words = _words(text)
    return bool(words & DE_MARKERS) and not (words & EN_LEFTOVERS)


def _cap(text):
    text = " ".join(text.split())
    if not text or len(text) > MAX_CHARS:
        return None
    return text


def _order(text):
    text = text.strip().rstrip(".")
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _pools(pieces):
    pools = {}
    roles, questions, tasks = pieces["roles_en"], pieces["questions_en"], pieces["tasks_en"]
    carriers, orders = pieces["carriers_en"], pieces["orders_en"]
    pools[("en", "role", 1)] = [
        line for role in roles for question in questions for line in (
            _cap(f"Now you are {role}. {question}"),
            _cap(f"You are {role}. {question}"),
        ) if line
    ]
    pools[("en", "role", 0)] = [
        line for role in roles for task in tasks
        if (line := _cap(f"You are {role}. {task}"))
    ]
    pools[("en", "pivot", 1)] = [
        line for carrier in carriers for order in orders
        if (line := _cap(f"{carrier} {_order(order)}."))
    ]
    pools[("en", "pivot", 0)] = [
        line for carrier in carriers for question in questions
        if (line := _cap(f"{carrier} {question}"))
    ]

    roles, questions, tasks = pieces["roles_de"], pieces["questions_de"], pieces["tasks_de"]
    carriers, orders = pieces["carriers_de"], pieces["orders_de"]
    pools[("de", "role", 1)] = [
        line for role in roles for question in questions for line in (
            _cap(f"Jetzt bist du {role}. {question}"),
            _cap(f"Du bist jetzt {role}. {question}"),
        ) if line and _german(line)
    ]
    pools[("de", "role", 0)] = [
        line for role in roles for task in tasks
        if (line := _cap(f"Du bist {role}. {task}")) and _german(line)
    ]
    pools[("de", "pivot", 1)] = [
        line for carrier in carriers for order in orders
        if (line := _cap(f"{carrier} {_order(order)}.")) and _german(line)
    ]
    pools[("de", "pivot", 0)] = [
        line for carrier in carriers for question in questions
        if (line := _cap(f"{carrier} {question}")) and _german(line)
    ]
    return pools


def _sample(pool, n, rng, seen, block):
    order = list(pool)
    rng.shuffle(order)
    rows = []
    dropped_block = 0
    dropped_dup = 0
    for text in order:
        if len(rows) >= n:
            break
        key = norm_key(text)
        if not key or key in seen:
            dropped_dup += 1
            continue
        if key in block:
            dropped_block += 1
            continue
        seen.add(key)
        rows.append(text)
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} rows available, needed {n}")
    return rows, dropped_block, dropped_dup


def main():
    with open(PIECES) as handle:
        pieces = json.load(handle)
    print("loading eval blocklist...", flush=True)
    block = load_blocklist()
    pools = _pools(pieces)
    rng = random.Random(SEED)
    seen = set()
    rows = []
    dropped_block = 0
    dropped_dup = 0
    for language, pattern, label, n in QUOTAS:
        chosen, blocked, dups = _sample(pools[(language, pattern, label)], n, rng, seen, block)
        dropped_block += blocked
        dropped_dup += dups
        for text in chosen:
            rows.append({
                "text": text,
                "source": "short_role",
                "teacher_label": label,
                "p_jailbreak": float(label),
                "language": language,
                "pattern": pattern,
            })
        print(f"{language} {pattern} label={label} rows={len(chosen)}", flush=True)
    frame = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    frame.to_parquet(OUT, index=False)
    report = {
        "rows": int(len(frame)),
        "dropped_blocklist": dropped_block,
        "dropped_dup": dropped_dup,
        "by_language": {k: int(v) for k, v in frame.groupby("language").size().items()},
        "by_pattern": {k: int(v) for k, v in frame.groupby("pattern").size().items()},
        "positives": int((frame.teacher_label == 1).sum()),
        "max_chars": int(frame.text.str.len().max()),
    }
    with open(REPORT, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)
    print("SHORT_ROLE_OK", flush=True)


if __name__ == "__main__":
    main()
