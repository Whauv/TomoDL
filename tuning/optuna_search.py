"""Optuna hyperparameter search for TomoDL."""

from __future__ import annotations

import copy
from typing import Any, Dict

import optuna
import torch

from training.finetune import train_and_validate_once


def _objective(trial: optuna.Trial, cfg: Dict[str, Any], device: torch.device, pretrained_ckpt: str | None) -> float:
    trial_cfg = copy.deepcopy(cfg)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    patch_size = trial.suggest_categorical("patch_size", [64, 80, 96, 112])
    w1 = trial.suggest_float("w1_seg", 0.1, 1.0)
    w2 = trial.suggest_float("w2_reg", 0.1, 1.0)

    trial_cfg["training"]["stage2"]["lr"] = lr
    trial_cfg["data"]["patch_size"] = [patch_size, patch_size, patch_size]
    trial_cfg["training"]["stage2"]["loss_weights"]["w1_seg"] = w1
    trial_cfg["training"]["stage2"]["loss_weights"]["w2_reg"] = w2

    f2 = train_and_validate_once(
        trial_cfg,
        device=device,
        pretrained_ckpt=pretrained_ckpt,
        max_epochs=int(cfg["tuning"]["max_epochs_per_trial"]),
    )
    trial.report(f2, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return f2


def run_optuna_search(cfg: Dict[str, Any], device: torch.device, pretrained_ckpt: str | None = None) -> optuna.Study:
    """Run Bayesian optimization with median pruning."""
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: _objective(trial, cfg, device, pretrained_ckpt),
        n_trials=int(cfg["tuning"]["n_trials"]),
        timeout=cfg["tuning"]["timeout_sec"],
    )
    return study
