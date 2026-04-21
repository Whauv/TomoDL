"""Convert extracted Kaggle JPG slice stacks to NPY volumes and generate manifests."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


ID_CANDIDATES = ["tomo_id", "tomogram_id", "id", "tomo", "name"]
Z_CANDIDATES = ["z", "motor_axis_0", "axis0"]
Y_CANDIDATES = ["y", "motor_axis_1", "axis1"]
X_CANDIDATES = ["x", "motor_axis_2", "axis2"]


def _norm_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col_map = {_norm_col(c): c for c in df.columns}
    for candidate in candidates:
        if candidate in col_map:
            return col_map[candidate]
    return None


def _to_manifest_path(path: Path, project_root: Path) -> str:
    rel = path.resolve().relative_to(project_root.resolve())
    return f"./{str(rel).replace('\\', '/')}"


def _load_volume_from_jpg_stack(
    tomo_dir: Path,
    slice_step: int = 1,
    resize_to: Optional[int] = None,
    output_dtype: str = "float32",
) -> np.ndarray:
    slices = sorted(tomo_dir.glob("slice_*.jpg"))[::max(1, slice_step)]
    if not slices:
        raise FileNotFoundError(f"No slice_*.jpg files found in {tomo_dir}")
    arrs = []
    for s in slices:
        img = Image.open(s).convert("L")
        if resize_to is not None and resize_to > 0:
            img = img.resize((resize_to, resize_to), resample=Image.BILINEAR)
        arrs.append(np.asarray(img, dtype=np.float32))
    volume = np.stack(arrs, axis=0)
    if output_dtype == "float16":
        volume = volume.astype(np.float16)
    return volume


def convert_split_to_npy(
    split_dir: Path,
    out_dir: Path,
    max_tomos: Optional[int] = None,
    slice_step: int = 1,
    resize_to: Optional[int] = None,
    output_dtype: str = "float32",
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Path] = {}
    tomo_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("tomo_")])
    if max_tomos is not None and max_tomos > 0:
        tomo_dirs = tomo_dirs[:max_tomos]
    for tomo in tqdm(tomo_dirs, desc=f"Converting {split_dir.name}", leave=False):
        tomo_id = tomo.name
        out_path = out_dir / f"{tomo_id}.npy"
        if not out_path.exists():
            vol = _load_volume_from_jpg_stack(
                tomo,
                slice_step=slice_step,
                resize_to=resize_to,
                output_dtype=output_dtype,
            )
            np.save(out_path, vol)
        mapping[tomo_id] = out_path
    return mapping


def build_manifests(
    project_root: Path,
    train_volume_map: Dict[str, Path],
    test_volume_map: Dict[str, Path],
    val_ratio: float,
    seed: int,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    labels_path = project_root / "train_labels.csv"
    sub_path = project_root / "sample_submission.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")
    if not sub_path.exists():
        raise FileNotFoundError(f"Missing {sub_path}")

    labels_df = pd.read_csv(labels_path)
    id_col = _find_col(labels_df, ID_CANDIDATES)
    z_col = _find_col(labels_df, Z_CANDIDATES)
    y_col = _find_col(labels_df, Y_CANDIDATES)
    x_col = _find_col(labels_df, X_CANDIDATES)
    if not id_col or not z_col or not y_col or not x_col:
        raise ValueError("train_labels.csv is missing tomo id / coordinate columns.")

    rows = []
    work = pd.DataFrame(
        {
            "tomo_id": labels_df[id_col].astype(str),
            "z": pd.to_numeric(labels_df[z_col], errors="coerce"),
            "y": pd.to_numeric(labels_df[y_col], errors="coerce"),
            "x": pd.to_numeric(labels_df[x_col], errors="coerce"),
        }
    )
    for tomo_id, chunk in work.groupby("tomo_id", sort=False):
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
    train_all = pd.DataFrame(rows)
    train_all["tomo_path"] = train_all["tomo_id"].map(train_volume_map)
    train_all = train_all[train_all["tomo_path"].notnull()].copy()
    train_all["tomo_path"] = train_all["tomo_path"].map(lambda p: _to_manifest_path(Path(p), project_root))

    stratify = train_all["has_motor"] if train_all["has_motor"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        train_all,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    sub_df = pd.read_csv(sub_path)
    sub_id_col = _find_col(sub_df, ID_CANDIDATES) or sub_df.columns[0]
    test_df = pd.DataFrame({"tomo_id": sub_df[sub_id_col].astype(str)})
    test_df["tomo_path"] = test_df["tomo_id"].map(test_volume_map)
    test_df = test_df[test_df["tomo_path"].notnull()].copy()
    test_df["tomo_path"] = test_df["tomo_path"].map(lambda p: _to_manifest_path(Path(p), project_root))

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    train_df[["tomo_id", "tomo_path", "has_motor", "z", "y", "x"]].sort_values("tomo_id").to_csv(train_path, index=False)
    val_df[["tomo_id", "tomo_path", "has_motor", "z", "y", "x"]].sort_values("tomo_id").to_csv(val_path, index=False)
    test_df[["tomo_id", "tomo_path"]].sort_values("tomo_id").to_csv(test_path, index=False)
    return train_path, val_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare extracted Kaggle JPG stacks for TomoDL")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-manifest-dir", type=str, default="./data")
    parser.add_argument("--volume-out-dir", type=str, default="./data/volumes")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-tomos", type=int, default=0, help="Use first N training tomograms (0 = all).")
    parser.add_argument("--max-test-tomos", type=int, default=0, help="Use first N test tomograms (0 = all).")
    parser.add_argument("--slice-step", type=int, default=1, help="Take every Kth slice (1 = all slices).")
    parser.add_argument("--resize-to", type=int, default=0, help="Resize each slice to NxN (0 = original).")
    parser.add_argument(
        "--output-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Saved numpy volume dtype.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    train_dir = project_root / "train"
    test_dir = project_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected extracted folders ./train and ./test at project root.")

    volume_out = Path(args.volume_out_dir).resolve()
    train_map = convert_split_to_npy(
        train_dir,
        volume_out / "train",
        max_tomos=(args.max_train_tomos if args.max_train_tomos > 0 else None),
        slice_step=max(1, args.slice_step),
        resize_to=(args.resize_to if args.resize_to > 0 else None),
        output_dtype=args.output_dtype,
    )
    test_map = convert_split_to_npy(
        test_dir,
        volume_out / "test",
        max_tomos=(args.max_test_tomos if args.max_test_tomos > 0 else None),
        slice_step=max(1, args.slice_step),
        resize_to=(args.resize_to if args.resize_to > 0 else None),
        output_dtype=args.output_dtype,
    )
    train_csv, val_csv, test_csv = build_manifests(
        project_root=project_root,
        train_volume_map=train_map,
        test_volume_map=test_map,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        output_dir=Path(args.output_manifest_dir).resolve(),
    )
    print(f"Generated: {train_csv}")
    print(f"Generated: {val_csv}")
    print(f"Generated: {test_csv}")


if __name__ == "__main__":
    main()
