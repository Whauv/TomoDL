"""Pre-submit dry run and strict validation for TomoDL submissions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.config import load_yaml_config
from evaluation.submission_validator import validate_submission_csv, validate_submission_df
from predict import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-submit validation for submission.csv")
    parser.add_argument("--config", type=str, default="./configs/config_predict_ultralite.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-inference", action="store_true", help="Run inference before validation.")
    parser.add_argument("--submission", type=str, default=None, help="Override submission file path.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    test_df = pd.read_csv(cfg["paths"]["test_csv"])
    expected_ids = [str(x) for x in test_df["tomo_id"].tolist()]

    submission_path = Path(args.submission or cfg["paths"]["submission_path"])

    if args.run_inference:
        ckpt = str(args.checkpoint or cfg["paths"]["finetune_ckpt"])
        pred_df = run_inference(cfg, ckpt)
        validate_submission_df(pred_df, expected_tomo_ids=expected_ids)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(submission_path, index=False)
        print(f"Dry-run inference + validation passed. Wrote: {submission_path}")
        return

    if not submission_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_path}")
    validate_submission_csv(str(submission_path), expected_tomo_ids=expected_ids)
    print(f"Submission validation passed: {submission_path}")


if __name__ == "__main__":
    main()
