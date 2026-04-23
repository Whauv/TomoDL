"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from core.errors import ConfigurationError


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Config file does not exist: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigurationError(f"Config root must be a dictionary: {cfg_path}")
    return cfg


def require_nested_keys(cfg: dict[str, Any], dotted_keys: Iterable[str]) -> None:
    """Validate that required dotted-path keys exist in a config dictionary."""
    for dotted_key in dotted_keys:
        node: Any = cfg
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigurationError(f"Missing required config key: {dotted_key}")
            node = node[part]
