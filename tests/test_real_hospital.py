"""Automated test for Out-of-Distribution Real Hospital Dataset Evaluation.

Verifies that models trained exclusively on synthetic data transfer smoothly to
realistic_hospital_testing_dataset_v8.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

from qms.config import load_config, abspath
from qms.data.prepare import prepare_real_hospital


def test_prepare_real_hospital_dataset():
    cfg = load_config()
    df_real = prepare_real_hospital(cfg)
    assert df_real is not None
    assert len(df_real) == 2000
    assert "no_show" in df_real.columns
    assert "service_time" in df_real.columns
    assert "waiting_time" in df_real.columns
    assert "arrival_delay" in df_real.columns


def test_real_hospital_metrics_exist_and_valid():
    cfg = load_config()
    metrics_path = abspath(cfg["reports"]["metrics_dir"]) / "real_hospital_test_metrics.json"
    comp_path = abspath(cfg["reports"]["metrics_dir"]) / "synthetic_vs_real_comparison.json"

    assert metrics_path.exists(), "real_hospital_test_metrics.json must exist"
    assert comp_path.exists(), "synthetic_vs_real_comparison.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["n_samples"] == 2000
    assert "no_show" in data["models"]
    assert "service_time" in data["models"]
    assert data["models"]["service_time"]["mae"] < 5.0  # within 5 mins error
    assert data["conformal_prediction"]["waiting_time"]["empirical_coverage_on_real"] >= 0.90
    assert data["fairness_audit"]["four_fifths_rule_satisfied"] is True
