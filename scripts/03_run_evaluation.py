#!/usr/bin/env python
"""Step 3 - run evaluation.

Compares the scheduling policies (FCFS / Static-Priority / Multi-Objective) on
the real attended-patient data, runs the no-show-aware buffer analysis, and
writes reports/metrics/policy_comparison.* + buffer_analysis.json + figures.

Usage:  python scripts/03_run_evaluation.py
"""
import _bootstrap  # noqa: F401

from qms.config import load_config
from qms.evaluation.evaluate import main as eval_main


def main() -> None:
    print("== Step 3/3: evaluating policies ==")
    cfg = load_config()
    eval_main(cfg)
    print("done.")


if __name__ == "__main__":
    main()
