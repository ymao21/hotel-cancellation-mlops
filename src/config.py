"""Load the project config and resolve paths."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config.

    Order: explicit `path` -> HOTEL_CONFIG env var -> config/config.yaml.
    """
    path = Path(path or os.environ.get("HOTEL_CONFIG", DEFAULT_CONFIG))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Return cfg['paths'][key] as an absolute Path (relative -> project root)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p
