"""Create deterministic fold manifests for reproducible evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic folds from train.csv")
    parser.add_argument("--train-csv", type=str, default="./data/train.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="./data/folds")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    required = {"tomo_id", "tomo_path", "has_motor", "z", "y", "x"}
    if not required.issubset(df.columns):
        raise ValueError(f"train csv missing columns: {sorted(required - set(df.columns))}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    y = df["has_motor"].astype(int).to_numpy()
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df, y), start=1):
        tr = df.iloc[tr_idx].copy().sort_values("tomo_id")
        va = df.iloc[va_idx].copy().sort_values("tomo_id")
        tr_path = out_dir / f"fold_{fold_idx}_train.csv"
        va_path = out_dir / f"fold_{fold_idx}_val.csv"
        tr.to_csv(tr_path, index=False)
        va.to_csv(va_path, index=False)
        print(f"Generated {tr_path} and {va_path}")


if __name__ == "__main__":
    main()
