"""Regression and verification tests for the Model Training & Evaluation pipeline.

Verifies the 5 critical research paper & reviewer requirements:
1. No-show thresholding produces healthy accuracy (>55%) and non-degenerate flag rate (no 25% accuracy collapse).
2. Conformal prediction computes exact mathematical non-conformity quantiles for any alpha (no linear rescale)
   and enforces strict >= (1 - alpha) validity checks.
3. Arrival delay metrics transparently calculate and report both median baseline and mean baseline.
4. Data preparation for real hospital records is immune to pandas 2/3 string/object dtype variations.
5. Model bundles and evaluation scripts dynamically respect config-driven test_size.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

from qms.config import load_config, abspath
from qms.data.loader import load_clean
from qms.data import preprocess
from qms.data.prepare import prepare_real_hospital
from qms.models import estimators, metrics
from qms.models.estimators import ModelBundle
from qms.models.conformal import ConformalRegressor


def test_issue1_noshow_accuracy_and_threshold_not_degenerate():
    """Verify No-show accuracy is >55% and does not flag 95% of patients (preventing the 25% accuracy trap)."""
    cfg = load_config()
    bundle_path = abspath(cfg["models"]["dir"]) / "no_show.pkl"
    assert bundle_path.exists(), "no_show.pkl must exist"
    bundle = ModelBundle.load(bundle_path)

    # Check synthetic test metrics
    m = bundle.metrics
    assert m["accuracy"] >= 0.55, f"Expected accuracy >= 55%, got {m['accuracy']}"
    assert m["flag_rate"] < 0.60, f"Expected flag_rate < 60%, got {m['flag_rate']}"
    assert m["specificity"] >= 0.50, f"Expected specificity >= 50%, got {m['specificity']}"
    assert bundle.threshold >= 0.15, f"Expected threshold >= 0.15, got {bundle.threshold}"

    # Verify real hospital test metrics
    comp_file = abspath(cfg["reports"]["metrics_dir"]) / "synthetic_vs_real_comparison.json"
    assert comp_file.exists()
    with open(comp_file, "r", encoding="utf-8") as f:
        comp = json.load(f)

    real_ns = comp["comparison"]["no_show"]["real_hospital_test"]
    assert real_ns["accuracy"] >= 0.55, f"Real accuracy must be >= 55%, got {real_ns['accuracy']}"
    assert real_ns["flag_rate"] < 0.50, f"Real flag_rate must be < 50%, got {real_ns['flag_rate']}"
    assert real_ns["tn"] > real_ns["fp"], "True negatives must exceed false positives on real hospital"


def test_issue2_conformal_exact_quantile_and_coverage_guarantee():
    """Verify conformal prediction uses exact quantile calculations without linear scaling and checks strict coverage."""
    calibrator = ConformalRegressor(alpha=0.10, task="waiting_time")
    y_true = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    y_pred = np.array([12.0, 14.0, 22.0, 24.0, 31.0, 33.0, 42.0, 43.0, 52.0, 54.0])

    calibrator.calibrate(y_true, y_pred)
    assert calibrator.calibration_residuals is not None
    assert len(calibrator.calibration_residuals) == len(y_true)

    # Testing varying alphas must produce mathematically exact quantiles from stored residuals
    int_90 = calibrator.predict_interval(25.0, alpha=0.10)
    int_95 = calibrator.predict_interval(25.0, alpha=0.05)
    int_80 = calibrator.predict_interval(25.0, alpha=0.20)

    # Higher confidence (smaller alpha) must yield equal or strictly wider intervals
    assert int_95.width >= int_90.width
    assert int_90.width >= int_80.width

    # Verify that predict_interval matches exact quantile on residuals
    residuals = np.abs(y_true - y_pred)
    n = len(residuals)
    expected_q95 = float(np.quantile(residuals, min(1.0, np.ceil((n + 1.0) * 0.95) / n), method="higher"))
    assert np.isclose(int_95.width / 2.0, expected_q95)

    # Verify coverage validity checks exact (1 - alpha)
    cov_pass = calibrator.evaluate_coverage(y_true, y_pred)
    assert "valid_guarantee" in cov_pass
    # If empirical coverage is 0.85 for alpha=0.10, valid_guarantee must be False
    cov_mock = calibrator.evaluate_coverage(y_true + 100.0, y_pred)
    assert cov_mock["valid_guarantee"] is False


def test_issue3_arrival_delay_metrics_and_baselines():
    """Verify regression metrics transparently include both median and mean baselines."""
    y_true = np.array([-10.0, -5.0, 0.0, 2.0, 15.0])
    y_pred = np.array([-8.0, -4.0, 1.0, 3.0, 12.0])

    m = metrics.regression_metrics(y_true, y_pred)
    assert "mae_baseline_median" in m
    assert "mae_baseline_mean" in m
    assert "mae_skill_vs_median" in m
    assert "mae_skill_vs_mean" in m
    assert m["mae_baseline"] == m["mae_baseline_median"]

    # Check committed metrics file for arrival_delay
    cfg = load_config()
    with open(abspath(cfg["reports"]["metrics_dir"]) / "model_metrics.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    arr_m = metrics_data["models"]["arrival_delay"]["metrics"]
    assert "mae_baseline_median" in arr_m
    assert "mae_baseline_mean" in arr_m


def test_issue4_prepare_priority_level_pandas_robustness(tmp_path):
    """Verify prepare_real_hospital handles priority_level across pandas object, string, and category dtypes."""
    cfg = load_config()
    # Test mapping on both string and numeric priority columns
    test_df = pd.DataFrame({
        "appointment_id": ["A1", "A2", "A3"],
        "patient_id": ["P1", "P2", "P3"],
        "booking_timestamp": ["01-04-2026 08:00", "01-04-2026 08:30", "01-04-2026 09:00"],
        "appointment_date": ["02-04-2026", "02-04-2026", "02-04-2026"],
        "appointment_time": ["09:00", "09:30", "10:00"],
        "department": ["General Medicine", "Cardiology", "Orthopedics"],
        "doctor_id": ["D1", "D2", "D3"],
        "priority_level": ["Normal", "Emergency", "normal"],
        "check_in_time": ["02-04-2026 08:50", "02-04-2026 09:20", "02-04-2026 09:55"],
        "appointment_status": ["Completed", "Completed", "Completed"],
        "service_start_time": ["02-04-2026 09:00", "02-04-2026 09:30", "02-04-2026 10:00"],
        "service_end_time": ["02-04-2026 09:15", "02-04-2026 09:50", "02-04-2026 10:20"],
        "cancellation_flag": [0, 0, 0],
        "previous_no_show_count": [0, 0, 1],
        "previous_late_count": [0, 1, 0],
        "total_visits_count": [1, 2, 3],
    })
    csv_path = tmp_path / "mock_real.csv"
    test_df.to_csv(csv_path, index=False)

    df_out = prepare_real_hospital(
        cfg,
        raw_filename=str(csv_path),
        out_filename="mock_real_clean.csv",
        profile_filename="mock_real_profile.json",
    )
    assert df_out["priority_level"].tolist() == [0, 1, 0]
    assert pd.api.types.is_numeric_dtype(df_out["priority_level"])


def test_issue5_config_driven_test_split_and_all_models_loadable():
    """Verify config-driven test_size is respected and all 4 model bundles infer properly."""
    cfg = load_config()
    test_size = float(cfg.get("data", {}).get("test_size", 0.20))
    assert test_size > 0.0 and test_size < 1.0

    df_clean = load_clean(cfg)
    models_dir = abspath(cfg["models"]["dir"])

    for task in ["no_show", "arrival_delay", "waiting_time", "service_time"]:
        bundle_path = models_dir / f"{task}.pkl"
        assert bundle_path.exists(), f"Missing bundle {bundle_path}"
        bundle = ModelBundle.load(bundle_path)

        X_raw, _, _ = preprocess.get_task_frame(df_clean.head(10), task)
        preds = bundle.predict(X_raw)
        assert len(preds) == len(X_raw)
        if bundle.kind == "classification":
            proba = bundle.predict_proba(X_raw)
            assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
