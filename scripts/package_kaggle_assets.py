"""Package checkpoint, config, and validators for Kaggle offline submission."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILES_TO_INCLUDE = [
    "predict.py",
    "scripts/kaggle_notebook_inference.py",
    "evaluation/submission_validator.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Kaggle inference assets")
    parser.add_argument("--checkpoint", type=str, default="./outputs_laptop/finetune_multitask.pt")
    parser.add_argument("--config", type=str, default="./configs/config_predict_ultralite.yaml")
    parser.add_argument("--out-dir", type=str, default="./outputs_laptop/kaggle_bundle")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    cfg = Path(args.config)
    if not ckpt.exists() or not cfg.exists():
        raise FileNotFoundError("checkpoint/config not found")

    shutil.copy2(ckpt, out_dir / ckpt.name)
    shutil.copy2(cfg, out_dir / cfg.name)

    for rel in FILES_TO_INCLUDE:
        src = Path(rel)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    archive_path = shutil.make_archive(str(out_dir), "zip", root_dir=str(out_dir))
    print(f"Created Kaggle bundle: {archive_path}")


if __name__ == "__main__":
    main()
