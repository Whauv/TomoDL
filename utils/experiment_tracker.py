"""Lightweight experiment tracking utilities (JSONL + artifacts)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentTracker:
    root_dir: str
    experiment_name: str

    def __post_init__(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.root_dir) / f"{self.experiment_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.meta_file = self.run_dir / "meta.json"

    def log_config(self, config: dict[str, Any]) -> None:
        self.meta_file.write_text(json.dumps({"config": config}, indent=2), encoding="utf-8")

    def log_metric(self, step: int, name: str, value: float) -> None:
        payload = {"step": int(step), "name": str(name), "value": float(value)}
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def log_artifact(self, file_path: str) -> str:
        src = Path(file_path)
        dst = self.run_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return str(dst)
