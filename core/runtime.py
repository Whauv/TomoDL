"""Runtime utilities shared by CLI entrypoints."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_reproducible_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    """Resolve the preferred torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir(path: str) -> None:
    """Create output directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)
