"""CLI to validate train/val/test manifests and estimate disk usage."""

from __future__ import annotations

import argparse
import json

from data.manifest_checks import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifest integrity")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--train-csv", type=str, default="./data/train.csv")
    parser.add_argument("--val-csv", type=str, default="./data/val.csv")
    parser.add_argument("--test-csv", type=str, default="./data/test.csv")
    args = parser.parse_args()

    report = {
        "train": validate_manifest(args.train_csv, project_root=args.project_root, is_test=False),
        "val": validate_manifest(args.val_csv, project_root=args.project_root, is_test=False),
        "test": validate_manifest(args.test_csv, project_root=args.project_root, is_test=True),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
