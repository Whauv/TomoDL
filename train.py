"""Main entry point for TomoDL training."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from ablation.ablation_study import plot_ablation, run_ablation
from training.finetune import run_finetuning
from training.pretrain import run_pretraining
from tuning.optuna_search import run_optuna_search


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run selected training mode."""
    parser = argparse.ArgumentParser(description="TomoDL trainer")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune", "both"], default="both")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search.")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    pretrained_ckpt = None
    if args.stage in {"pretrain", "both"}:
        pretrained_ckpt = run_pretraining(cfg, device)

    if args.optuna:
        study = run_optuna_search(cfg, device, pretrained_ckpt=pretrained_ckpt)
        print("Best trial:", study.best_trial.number)
        print("Best params:", study.best_params)
        print("Best F2:", study.best_value)

    if args.stage in {"finetune", "both"}:
        run_finetuning(cfg, device, pretrained_ckpt=pretrained_ckpt)

    if args.ablation:
        results = run_ablation(cfg, device)
        out_plot = str(Path(cfg["paths"]["output_dir"]) / "ablation_f2.png")
        plot_ablation(results, out_plot)
        print("Ablation results:", results)
        print("Plot saved to:", out_plot)


if __name__ == "__main__":
    main()
