"""Inference CLI for Kaggle submission generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import load_yaml_config, require_nested_keys
from core.errors import cli_entrypoint
from core.runtime import ensure_output_dir, resolve_device
from evaluation.submission_validator import validate_submission_df
from inference.pipeline import (
    HeatmapPredictorFactory,
    SUBMISSION_COLUMNS,
    load_inference_manifest,
    predict_submission_rows,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TomoDL inference")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser


def _resolve_checkpoint(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    return str(args.checkpoint or cfg["paths"]["finetune_ckpt"])


def _validate_predict_config(cfg: dict[str, Any]) -> None:
    require_nested_keys(
        cfg,
        [
            "paths.test_csv",
            "paths.submission_path",
            "paths.finetune_ckpt",
            "data.patch_size",
            "data.normalization",
            "evaluation.tta_rotations",
            "evaluation.tta_policy",
            "evaluation.ensemble_weights.unet3d",
            "evaluation.ensemble_weights.resnet2d",
            "evaluation.ensemble_weights.detr3d",
            "inference.no_motor_threshold",
            "inference.low_memory_mode",
            "inference.sliding_window_if_large",
            "inference.window_overlap",
            "inference.instance_threshold",
            "inference.instance_min_size",
            "inference.instance_nms_distance",
            "model",
        ],
    )


def run_inference(cfg: dict[str, Any], ckpt_path: str) -> pd.DataFrame:
    """Run model inference and build the submission dataframe."""
    device = resolve_device()
    test_df = load_inference_manifest(str(cfg["paths"]["test_csv"]))
    predictor = HeatmapPredictorFactory(cfg=cfg, ckpt_path=ckpt_path, device=device)
    rows = predict_submission_rows(test_df=test_df, cfg=cfg, predictor=predictor, device=device)
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


@cli_entrypoint
def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_yaml_config(args.config)
    _validate_predict_config(cfg)

    checkpoint_path = _resolve_checkpoint(args, cfg)
    submission = run_inference(cfg, checkpoint_path)
    expected_ids = load_inference_manifest(str(cfg["paths"]["test_csv"]))["tomo_id"].astype(str).tolist()
    validate_submission_df(submission, expected_tomo_ids=expected_ids)
    out_path = Path(cfg["paths"]["submission_path"])
    ensure_output_dir(str(out_path.parent))
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
