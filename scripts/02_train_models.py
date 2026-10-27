#!/usr/bin/env python
"""Step 2 - train models.

Trains the three predictive models (no-show, arrival-delay, waiting-time) on the
cleaned data with a temporal train/test split, saves the model bundles to
models/, and writes reports/metrics/model_metrics.json plus figures.

Usage:  python scripts/02_train_models.py
"""
import _bootstrap  # noqa: F401

from qms.config import load_config
from qms.models.train_all import main as train_main


def main() -> None:
    print("== Step 2/3: training models ==")
    cfg = load_config()
    train_main(cfg)
    print("done.")


if __name__ == "__main__":
    main()
