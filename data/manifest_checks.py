"""Manifest integrity checks and disk guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_TRAIN_COLUMNS = {"tomo_id", "tomo_path", "has_motor", "z", "y", "x"}
REQUIRED_TEST_COLUMNS = {"tomo_id", "tomo_path"}


def _resolve_project_path(project_root: Path, path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    cleaned = str(path_like).replace("./", "", 1)
    return (project_root / cleaned).resolve()


def validate_manifest(csv_path: str, project_root: str = ".", is_test: bool = False) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    expected = REQUIRED_TEST_COLUMNS if is_test else REQUIRED_TRAIN_COLUMNS
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    root = Path(project_root).resolve()
    missing_files = 0
    bad_coordinates = 0
    total_bytes = 0

    for row in df.itertuples(index=False):
        tomo_path = _resolve_project_path(root, str(row.tomo_path))
        if not tomo_path.exists():
            missing_files += 1
            continue
        total_bytes += int(tomo_path.stat().st_size)
        if not is_test and int(row.has_motor) == 1:
            coords = np.asarray([row.z, row.y, row.x], dtype=np.float64)
            if not np.all(np.isfinite(coords)) or np.any(coords < 0):
                bad_coordinates += 1

    return {
        "rows": int(len(df)),
        "missing_files": int(missing_files),
        "bad_coordinates": int(bad_coordinates),
        "total_bytes": int(total_bytes),
        "total_gb": float(total_bytes / (1024**3)),
    }


def enforce_disk_guardrail(estimated_bytes: int, min_free_gb: float, target_dir: str) -> None:
    target = Path(target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    free_bytes = 0
    try:
        import shutil

        usage = shutil.disk_usage(str(target))
        free_bytes = int(usage.free)
    except Exception:
        return

    min_free_bytes = int(float(min_free_gb) * (1024**3))
    if free_bytes - int(estimated_bytes) < min_free_bytes:
        raise RuntimeError(
            f"Disk guardrail hit: free={free_bytes/(1024**3):.2f} GB, "
            f"estimated_write={estimated_bytes/(1024**3):.2f} GB, min_free={min_free_gb:.2f} GB"
        )
