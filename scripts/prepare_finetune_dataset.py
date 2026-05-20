"""Prepare subject-split datalist JSON from manifest.csv for adapter fine-tuning."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",  default="D:/Dataset MRI/manifest.csv")
    p.add_argument("--out",       default="data/finetune_split.json")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac",   type=float, default=0.15)
    p.add_argument("--max_slices", type=int, default=2000)
    args = p.parse_args()
    assert args.train_frac + args.val_frac < 1.0, "train_frac + val_frac must be < 1.0"

    df = pd.read_csv(args.manifest).dropna(subset=["file_path", "anatomy"])
    # Composite subject key to avoid leakage across source datasets
    df["_subject_key"] = df["source_dataset"].astype(str) + "|" + df["subject_id"].astype(str)

    subjects = sorted(df["_subject_key"].unique())
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * args.train_frac)
    n_val   = int(n * args.val_frac)
    train_s = set(subjects[:n_train])
    val_s   = set(subjects[n_train:n_train + n_val])
    test_s  = set(subjects[n_train + n_val:])

    def to_records(subj_set, max_rows=None):
        rows = df[df["_subject_key"].isin(subj_set)].copy()
        if max_rows and len(rows) > max_rows:
            rows = rows.sample(max_rows, random_state=args.seed)
        cols = ["file_path", "source_dataset", "subject_id", "anatomy"]
        if "vendor" in df.columns:
            cols.append("vendor")
        rows = rows[cols]
        if "vendor" not in df.columns:
            rows = rows.assign(vendor=None)
        return rows.to_dict("records")

    split = {
        "train": to_records(train_s, args.max_slices),
        "val":   to_records(val_s),
        "test":  to_records(test_s),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(split, indent=2))
    print(f"Split: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test slices")
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
