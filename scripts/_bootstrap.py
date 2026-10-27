"""Shared bootstrap so scripts run without installing the package.

Adds ``src/`` to sys.path, enabling ``import qms.*`` from a plain checkout:

    python scripts/01_prepare_data.py

(For a proper install use ``pip install -e .`` and the bootstrap is a no-op.)
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
