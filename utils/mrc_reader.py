"""MRC volume reader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mrcfile
import numpy as np


def read_mrc(path: str) -> np.ndarray:
    """Read an .mrc file into a float32 NumPy array."""
    p = Path(path)
    if p.suffix.lower() != ".mrc":
        raise ValueError(f"Expected .mrc file, got: {path}")
    with mrcfile.open(p, permissive=True) as mrc:
        return np.asarray(mrc.data, dtype=np.float32)
