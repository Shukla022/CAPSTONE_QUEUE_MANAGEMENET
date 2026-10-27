"""Central configuration loader.

Resolves the project root and loads ``config/config.yaml``. All relative
paths in the config can be turned into absolute paths via :func:`abspath`.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import yaml

# src/qms/config.py -> parents[2] == project root
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    """Load the YAML config once and cache it."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def abspath(rel: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def ensure_dirs(*rels: str) -> None:
    """Create directories (config-relative) if they do not exist."""
    for r in rels:
        abspath(r).mkdir(parents=True, exist_ok=True)
