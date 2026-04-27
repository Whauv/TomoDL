"""Kaggle notebook-ready inference runner for TomoDL.

This script is intended to be copied into a Kaggle notebook and executed with
internet disabled. It writes `/kaggle/working/submission.csv` with the exact
competition-required columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.config import load_yaml_config
from predict import run_inference


def _build_kaggle_paths_cfg(base_cfg: dict[str, Any], dataset_root: str, checkpoint_path: str) -> dict[str, Any]:
    """Create Kaggle-runtime overrides without mutating the base config."""
    cfg = dict(base_cfg)
    cfg["paths"] = dict(base_cfg["paths"])
    cfg["paths"]["test_csv"] = str(Path(dataset_root) / "test.csv")
    cfg["paths"]["submission_path"] = "/kaggle/working/submission.csv"
    cfg["paths"]["finetune_ckpt"] = checkpoint_path
    return cfg


def main() -> None:
    # Update these two paths in your Kaggle notebook cell before running.
    config_path = "/kaggle/input/tomodl-config/config_predict_ultralite.yaml"
    dataset_root = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
    checkpoint_path = "/kaggle/input/tomodl-checkpoint/finetune_multitask.pt"

    cfg = load_yaml_config(config_path)
    cfg = _build_kaggle_paths_cfg(cfg, dataset_root=dataset_root, checkpoint_path=checkpoint_path)
    submission = run_inference(cfg, ckpt_path=cfg["paths"]["finetune_ckpt"])
    out_path = Path(cfg["paths"]["submission_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved Kaggle submission file: {out_path}")
    print("Columns:", list(submission.columns))
    print("Rows:", len(submission))


if __name__ == "__main__":
    main()
