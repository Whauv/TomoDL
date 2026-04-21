"""Prepare TomoDL manifests from BYU Kaggle competition files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


ID_CANDIDATES = ["tomo_id", "tomogram_id", "id", "tomo", "name"]
Z_CANDIDATES = ["z", "motor_axis_0", "motoraxis0", "axis0", "motor_z"]
Y_CANDIDATES = ["y", "motor_axis_1", "motoraxis1", "axis1", "motor_y"]
X_CANDIDATES = ["x", "motor_axis_2", "motoraxis2", "axis2", "motor_x"]


def _norm_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {_norm_col(c): c for c in df.columns}
    for candidate in candidates:
        if candidate in col_map:
            return col_map[candidate]
    return None


def _discover_volumes(root: Path) -> Dict[str, Path]:
    volumes: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mrc", ".npy"}:
            continue
        tomo_id = path.stem
        # keep shortest path for duplicate stems
        if tomo_id not in volumes or len(str(path)) < len(str(volumes[tomo_id])):
            volumes[tomo_id] = path
    return volumes


def _looks_like_label_csv(df: pd.DataFrame) -> bool:
    id_col = _find_col(df, ID_CANDIDATES)
    z_col = _find_col(df, Z_CANDIDATES)
    y_col = _find_col(df, Y_CANDIDATES)
    x_col = _find_col(df, X_CANDIDATES)
    return id_col is not None and z_col is not None and y_col is not None and x_col is not None


def _find_label_csv(root: Path) -> Path:
    candidates = sorted(root.rglob("*.csv"))
    for path in candidates:
        try:
            df = pd.read_csv(path, nrows=32)
            if _looks_like_label_csv(df):
                return path
        except Exception:
            continue
    raise FileNotFoundError(
        "Could not find a label CSV with tomo_id + coordinate columns in Kaggle data root."
    )


def _find_sample_submission_csv(root: Path) -> Optional[Path]:
    preferred_names = {"sample_submission.csv", "submission.csv", "sample.csv"}
    for path in root.rglob("*.csv"):
        if path.name.lower() in preferred_names:
            return path
    return None


def _to_manifest_path(volume_path: Path, project_root: Path) -> str:
    try:
        rel = volume_path.resolve().relative_to(project_root.resolve())
        return f"./{str(rel).replace('\\', '/')}"
    except Exception:
        return str(volume_path.resolve())


def _load_labels(label_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(label_csv)
    id_col = _find_col(df, ID_CANDIDATES)
    z_col = _find_col(df, Z_CANDIDATES)
    y_col = _find_col(df, Y_CANDIDATES)
    x_col = _find_col(df, X_CANDIDATES)
    if not id_col or not z_col or not y_col or not x_col:
        raise ValueError(
            f"Label CSV {label_csv} must contain id and 3 coordinate columns. "
            f"Found columns: {list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "tomo_id": df[id_col].astype(str),
            "z": pd.to_numeric(df[z_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "x": pd.to_numeric(df[x_col], errors="coerce"),
        }
    )
    # Aggregate duplicates by taking the first valid positive row.
    out = out.sort_values(["tomo_id"]).reset_index(drop=True)
    grouped = []
    for tomo_id, chunk in out.groupby("tomo_id", sort=False):
        valid = chunk.dropna(subset=["z", "y", "x"])
        if len(valid) == 0:
            grouped.append({"tomo_id": tomo_id, "z": 0.0, "y": 0.0, "x": 0.0, "has_motor": 0})
            continue
        row = valid.iloc[0]
        has_motor = int((row["z"] >= 0) and (row["y"] >= 0) and (row["x"] >= 0))
        grouped.append(
            {
                "tomo_id": tomo_id,
                "z": float(row["z"]) if has_motor else 0.0,
                "y": float(row["y"]) if has_motor else 0.0,
                "x": float(row["x"]) if has_motor else 0.0,
                "has_motor": has_motor,
            }
        )
    return pd.DataFrame(grouped)


def build_manifests(
    kaggle_root: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    project_root: Path,
) -> Tuple[Path, Path, Path]:
    """Build train/val/test CSV manifests expected by TomoDL."""
    volumes = _discover_volumes(kaggle_root)
    if not volumes:
        raise FileNotFoundError(
            f"No .mrc/.npy tomograms found under {kaggle_root}. "
            "Place Kaggle competition files in this directory first."
        )

    label_csv = _find_label_csv(kaggle_root)
    labels = _load_labels(label_csv)
    labels["tomo_path"] = labels["tomo_id"].map(lambda t: volumes.get(t))
    labels = labels[labels["tomo_path"].notnull()].copy()
    labels["tomo_path"] = labels["tomo_path"].map(lambda p: _to_manifest_path(Path(p), project_root))

    if labels.empty:
        raise RuntimeError("No labeled tomograms matched discovered volume files.")

    stratify = labels["has_motor"] if labels["has_motor"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    sample_submission = _find_sample_submission_csv(kaggle_root)
    if sample_submission is not None:
        sub_df = pd.read_csv(sample_submission)
        id_col = _find_col(sub_df, ID_CANDIDATES) or sub_df.columns[0]
        test_ids = sub_df[id_col].astype(str).tolist()
    else:
        labeled_ids = set(labels["tomo_id"].tolist())
        test_ids = [tid for tid in volumes.keys() if tid not in labeled_ids]
    test_df = pd.DataFrame({"tomo_id": test_ids})
    test_df["tomo_path"] = test_df["tomo_id"].map(lambda t: volumes.get(t))
    test_df = test_df[test_df["tomo_path"].notnull()].copy()
    test_df["tomo_path"] = test_df["tomo_path"].map(lambda p: _to_manifest_path(Path(p), project_root))

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df = train_df[["tomo_id", "tomo_path", "has_motor", "z", "y", "x"]].sort_values("tomo_id")
    val_df = val_df[["tomo_id", "tomo_path", "has_motor", "z", "y", "x"]].sort_values("tomo_id")
    test_df = test_df[["tomo_id", "tomo_path"]].sort_values("tomo_id")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, val_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TomoDL CSV manifests from Kaggle data.")
    parser.add_argument(
        "--kaggle-root",
        type=str,
        default="./data/kaggle/raw",
        help="Directory containing downloaded and extracted Kaggle competition files.",
    )
    parser.add_argument("--output-dir", type=str, default="./data", help="Directory to write train/val/test manifests.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_path, val_path, test_path = build_manifests(
        kaggle_root=Path(args.kaggle_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        project_root=project_root,
    )
    print(f"Generated: {train_path}")
    print(f"Generated: {val_path}")
    print(f"Generated: {test_path}")


if __name__ == "__main__":
    main()
