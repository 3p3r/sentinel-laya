"""Write data/en3k_raw.parquet from 3nesdeniz/english-prompt-injection-3k.

Uses the dataset's own 0/1 label. Does not call Sentinel.
"""
import pandas as pd
from datasets import load_dataset

OUT = "data/en3k_raw.parquet"
NAME = "3nesdeniz/english-prompt-injection-3k"


def _bit(row):
    label = row.get("label")
    if isinstance(label, str):
        return 1 if label.strip().lower() in {"1", "injection", "attack", "jailbreak"} else 0
    if label is None and str(row.get("class", "")).strip().lower() == "injection":
        return 1
    return int(label)


def main():
    ds = load_dataset(NAME)
    rows = []
    for split in ds:
        for row in ds[split]:
            bit = _bit(row)
            family = str(row.get("attack_family") or "row")
            rows.append({
                "text": str(row["text"]),
                "source": f"en3k_{family}",
                "p_jailbreak": float(bit),
                "teacher_label": bit,
            })
    out = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"rows={len(out)} pos={int(out.teacher_label.sum())} -> {OUT}", flush=True)
    print("EN3K_FETCH_OK", flush=True)


if __name__ == "__main__":
    main()
