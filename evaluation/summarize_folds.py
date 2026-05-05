"""Aggregate per-fold metric JSON files into mean/std summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = ["f2", "localization_at_10"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cross-validation metrics")
    parser.add_argument("--metrics-dir", type=str, default="./outputs_laptop/folds")
    parser.add_argument("--out", type=str, default="./outputs_laptop/fold_summary.json")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    files = sorted(metrics_dir.glob("fold_*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No fold metrics found in {metrics_dir}")

    rows = [json.loads(fp.read_text(encoding="utf-8")) for fp in files]
    summary = {"num_folds": len(rows), "folds": rows}

    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
