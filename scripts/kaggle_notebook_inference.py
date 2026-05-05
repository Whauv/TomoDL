"""Kaggle notebook-ready offline inference runner for TomoDL (deterministic, 12h-safe)."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.config import load_yaml_config
from evaluation.submission_validator import validate_submission_df
from predict import run_inference


def _build_kaggle_paths_cfg(base_cfg: dict[str, Any], dataset_root: str, checkpoint_path: str) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["paths"] = dict(base_cfg["paths"])
    cfg["paths"]["test_csv"] = str(Path(dataset_root) / "test.csv")
    cfg["paths"]["submission_path"] = "/kaggle/working/submission.csv"
    cfg["paths"]["finetune_ckpt"] = checkpoint_path

    # Deterministic and runtime-safe defaults.
    cfg.setdefault("inference", {})
    cfg["inference"]["batch_size"] = 1
    cfg["inference"]["low_memory_mode"] = True
    cfg["inference"]["sliding_window_if_large"] = True
    cfg["evaluation"]["tta_rotations"] = False
    return cfg


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _set_seed(42)

    config_path = "/kaggle/input/tomodl-config/config_predict_ultralite.yaml"
    dataset_root = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
    checkpoint_path = "/kaggle/input/tomodl-checkpoint/finetune_multitask.pt"

    cfg = load_yaml_config(config_path)
    cfg = _build_kaggle_paths_cfg(cfg, dataset_root=dataset_root, checkpoint_path=checkpoint_path)
    submission = run_inference(cfg, ckpt_path=cfg["paths"]["finetune_ckpt"])

    test_df = pd.read_csv(cfg["paths"]["test_csv"])
    validate_submission_df(submission, expected_tomo_ids=test_df["tomo_id"].astype(str).tolist())

    out_path = Path(cfg["paths"]["submission_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved Kaggle submission: {out_path}")
    print("Rows:", len(submission))


if __name__ == "__main__":
    main()
