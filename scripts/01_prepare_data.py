#!/usr/bin/env python
"""Step 1 - prepare data.

Reads data/raw/hospital_dataset_preprocessed.csv, validates + cleans it, and
writes data/processed/clean.csv plus a data_profile.json summary.

Usage:  python scripts/01_prepare_data.py
"""
import _bootstrap  # noqa: F401  (adds src/ to path)

from qms.config import load_config
from qms.data.prepare import clean


def main() -> None:
    print("== Step 1/3: preparing data ==")
    cfg = load_config()
    clean(cfg)
    print("done.")


if __name__ == "__main__":
    main()
