"""Prepare TomoDL manifests from Hugging Face mirror dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


ID_CANDIDATES = ["tomo_id", "tomogram_id", "id", "tomo", "name"]
Z_CANDIDATES = ["motor_axis_0", "z", "axis0"]
Y_CANDIDATES = ["motor_axis_1", "y", "axis1"]
X_CANDIDATES = ["motor_axis_2", "x", "axis2"]


def _norm_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {_norm_col(c): c for c in df.columns}
    for candidate in candidates:
        if candidate in col_map:
            return col_map[candidate]
    return None


def _to_manifest_path(volume_path: Path, project_root: Path) -> str:
    try:
        rel = volume_path.resolve().relative_to(project_root.resolve())
        return f"./{str(rel).replace('\\', '/')}"
    except Exception:
        return str(volume_path.resolve())


def _download_hf_dataset(repo_id: str, local_dir: Path) -> Path:
    if snapshot_download is None:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir


def _discover_volumes(root: Path) -> Dict[str, Path]:
    volumes: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".npy", ".mrc"}:
            continue
        tomo_id = path.stem
        if tomo_id not in volumes or len(str(path)) < len(str(volumes[tomo_id])):
            volumes[tomo_id] = path
    return volumes


def _find_labels_csv(root: Path) -> Path:
    candidates = sorted(root.rglob("*.csv"))
    for p in candidates:
        if p.name.lower() in {"train_labels.csv", "train_label.csv"}:
            return p
    # fallback by columns
    for p in candidates:
        try:
            df = pd.read_csv(p, nrows=16)
        except Exception:
            continue
        id_col = _find_col(df, ID_CANDIDATES)
        z_col = _find_col(df, Z_CANDIDATES)
        y_col = _find_col(df, Y_CANDIDATES)
        x_col = _find_col(df, X_CANDIDATES)
        if id_col and z_col and y_col and x_col:
            return p
    raise FileNotFoundError("Could not locate a train labels CSV in the HF mirror root.")


def _find_sample_submission(root: Path) -> Optional[Path]:
    for p in root.rglob("*.csv"):
        if p.name.lower() == "sample_submission.csv":
            return p
    return None


def _collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _find_col(df, ID_CANDIDATES)
    z_col = _find_col(df, Z_CANDIDATES)
    y_col = _find_col(df, Y_CANDIDATES)
    x_col = _find_col(df, X_CANDIDATES)
    if not id_col or not z_col or not y_col or not x_col:
        raise ValueError("train_labels CSV does not contain required columns for tomo_id and coordinates.")
    labels = pd.DataFrame(
        {
            "tomo_id": df[id_col].astype(str),
            "z": pd.to_numeric(df[z_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "x": pd.to_numeric(df[x_col], errors="coerce"),
        }
    )

    # Keep one label per tomogram for this project's current dataset contract.
    rows = []
    for tomo_id, chunk in labels.groupby("tomo_id", sort=False):
        valid = chunk.dropna(subset=["z", "y", "x"])
        if valid.empty:
            rows.append({"tomo_id": tomo_id, "has_motor": 0, "z": 0.0, "y": 0.0, "x": 0.0})
            continue
        row = valid.iloc[0]
        has_motor = int((row["z"] >= 0) and (row["y"] >= 0) and (row["x"] >= 0))
        rows.append(
            {
                "tomo_id": tomo_id,
                "has_motor": has_motor,
                "z": float(row["z"]) if has_motor else 0.0,
                "y": float(row["y"]) if has_motor else 0.0,
                "x": float(row["x"]) if has_motor else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_manifests(
    hf_root: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    project_root: Path,
) -> Tuple[Path, Path, Path]:
    volumes = _discover_volumes(hf_root)
    if not volumes:
        raise FileNotFoundError(f"No .npy/.mrc tomograms found under {hf_root}")

    labels_csv = _find_labels_csv(hf_root)
    labels_df = _collapse_labels(pd.read_csv(labels_csv))
    labels_df["tomo_path"] = labels_df["tomo_id"].map(volumes)
    labels_df = labels_df[labels_df["tomo_path"].notnull()].copy()
    if labels_df.empty:
        raise RuntimeError("No labeled tomo_ids matched discovered volumes.")
    labels_df["tomo_path"] = labels_df["tomo_path"].map(lambda p: _to_manifest_path(Path(p), project_root))

    stratify = labels_df["has_motor"] if labels_df["has_motor"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        labels_df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    sample_submission = _find_sample_submission(hf_root)
    if sample_submission is not None:
        sub_df = pd.read_csv(sample_submission)
        id_col = _find_col(sub_df, ID_CANDIDATES) or sub_df.columns[0]
        test_ids = sub_df[id_col].astype(str).tolist()
    else:
        labeled_ids = set(labels_df["tomo_id"].tolist())
        test_ids = [tid for tid in volumes.keys() if tid not in labeled_ids]
    test_df = pd.DataFrame({"tomo_id": test_ids})
    test_df["tomo_path"] = test_df["tomo_id"].map(volumes)
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
    parser = argparse.ArgumentParser(description="Prepare TomoDL CSV manifests from HF mirror dataset.")
    parser.add_argument("--repo-id", type=str, default="Floppanacci/tomogram-Bacterial-Flagellar-motors-location")
    parser.add_argument("--hf-root", type=str, default="./data/hf_mirror/raw")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset snapshot from HF into --hf-root before manifest generation.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    hf_root = Path(args.hf_root).resolve()
    if args.download:
        _download_hf_dataset(args.repo_id, hf_root)
    train_path, val_path, test_path = build_manifests(
        hf_root=hf_root,
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
