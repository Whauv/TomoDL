"""One-command reproducibility runner for ablations and report assets."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible TomoDL experiment bundle")
    parser.add_argument("--config", type=str, default="./configs/config_laptop_8gb.yaml")
    parser.add_argument("--output-dir", type=str, default="./outputs_laptop/repro")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        _run(["python", "./train.py", "--config", args.config, "--stage", "finetune"])

    _run(["python", "./scripts/pre_submit_check.py", "--config", "./configs/config_predict_ultralite.yaml", "--run-inference"])

    summary = {
        "config": args.config,
        "artifacts": [
            "outputs_laptop/finetune_multitask.pt",
            "outputs_laptop/submission.csv",
            "outputs_laptop/no_motor_calibration.json",
        ],
    }
    (out_dir / "repro_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "repro_summary.json")


if __name__ == "__main__":
    main()
