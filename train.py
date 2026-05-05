"""Training CLI for TomoDL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ablation.ablation_study import plot_ablation, run_ablation
from core.config import load_yaml_config, require_nested_keys
from core.errors import cli_entrypoint
from core.runtime import ensure_output_dir, resolve_device, set_reproducible_seed
from training.finetune import run_finetuning
from training.pretrain import run_pretraining
from tuning.optuna_search import run_optuna_search
from utils.cleanlab_filter import run_cleanlab_on_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TomoDL trainer")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune", "both"], default="both")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search.")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study.")
    parser.add_argument("--cleanlab", action="store_true", help="Run cleanlab label filtering before finetuning.")
    parser.add_argument("--cleanlab-mode", type=str, choices=["remove", "relabel"], default="remove")
    return parser


def _validate_train_config(cfg: dict[str, Any]) -> None:
    require_nested_keys(
        cfg,
        [
            "seed",
            "paths.output_dir",
            "paths.pretrain_ckpt",
            "paths.finetune_ckpt",
            "training.stage1",
            "training.stage2",
            "data",
            "model",
            "evaluation",
        ],
    )




def _run_optional_cleanlab(enable_cleanlab: bool, cleanlab_mode: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if not enable_cleanlab:
        return cfg
    out_dir = Path(cfg["paths"]["output_dir"]) / "cleanlab"
    review_path, cleaned_path = run_cleanlab_on_manifest(
        train_csv_path=str(cfg["paths"]["train_csv"]),
        output_dir=str(out_dir),
        mode=cleanlab_mode,
        issue_fraction=float(cfg.get("training", {}).get("cleanlab", {}).get("issue_fraction", 0.05)),
    )
    print(f"Cleanlab review file: {review_path}")
    print(f"Cleaned train manifest: {cleaned_path}")
    cfg2 = dict(cfg)
    paths = dict(cfg2["paths"])
    paths["train_csv"] = cleaned_path
    cfg2["paths"] = paths
    return cfg2


def _run_optional_pretraining(stage: str, cfg: dict[str, Any], device: torch.device) -> str | None:
    if stage in {"pretrain", "both"}:
        return run_pretraining(cfg, device)
    return None


def _run_optional_optuna(enable_optuna: bool, cfg: dict[str, Any], device: torch.device, pretrained_ckpt: str | None) -> None:
    if not enable_optuna:
        return
    study = run_optuna_search(cfg, device, pretrained_ckpt=pretrained_ckpt)
    print("Best trial:", study.best_trial.number)
    print("Best params:", study.best_params)
    print("Best F2:", study.best_value)


def _run_optional_finetuning(stage: str, cfg: dict[str, Any], device: torch.device, pretrained_ckpt: str | None) -> None:
    if stage in {"finetune", "both"}:
        run_finetuning(cfg, device, pretrained_ckpt=pretrained_ckpt)


def _run_optional_ablation(enable_ablation: bool, cfg: dict[str, Any], device: torch.device) -> None:
    if not enable_ablation:
        return
    results = run_ablation(cfg, device)
    output_plot = str(Path(cfg["paths"]["output_dir"]) / "ablation_f2.png")
    plot_ablation(results, output_plot)
    print("Ablation results:", results)
    print("Plot saved to:", output_plot)


@cli_entrypoint
def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    _validate_train_config(cfg)

    set_reproducible_seed(int(cfg["seed"]))
    device = resolve_device()
    ensure_output_dir(str(cfg["paths"]["output_dir"]))

    cfg = _run_optional_cleanlab(args.cleanlab, args.cleanlab_mode, cfg)
    pretrained_ckpt = _run_optional_pretraining(args.stage, cfg, device)
    _run_optional_optuna(args.optuna, cfg, device, pretrained_ckpt)
    _run_optional_finetuning(args.stage, cfg, device, pretrained_ckpt)
    _run_optional_ablation(args.ablation, cfg, device)


if __name__ == "__main__":
    main()
