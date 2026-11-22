#!/usr/bin/env python
"""Step 5 - Train on Synthetic Data, Evaluate on Real Hospital Dataset.

Research Protocol:
------------------
1. Training: Models are trained EXCLUSIVELY on the synthetic clinic dataset
   (data/processed/clean.csv - 20,000 synthetic records).
2. Out-of-Distribution Testing: Evaluated zero-shot on the real-world hospital dataset
   (data/processed/real_hospital_clean.csv - 2,000 real records from realistic_hospital_testing_dataset_v8.csv).
3. Evaluates:
   - Model generalization (no_show, arrival_delay, waiting_time, service_time)
   - Conformal Prediction coverage guarantees on real hospital patients
   - Dynamic overbooking buffer performance on real clinic slots
   - Discrete-event simulation policy comparisons (FCFS vs Static-Priority vs Multi-Objective)
   - Algorithmic fairness audit (Disparate Impact & Demographic Parity)
4. Persists comparative reports:
   - reports/metrics/real_hospital_test_metrics.json
   - reports/metrics/synthetic_vs_real_comparison.json
   - reports/figures/real_hospital_pred_vs_actual.png
   - reports/figures/synthetic_vs_real_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401 (adds src/ and app/ to path)

from qms.config import load_config, abspath, ensure_dirs
from qms.data.loader import load_clean
from qms.data import preprocess
from qms.data.prepare import prepare_real_hospital
from qms.models import estimators, metrics
from qms.models.estimators import ModelBundle
from qms.models.conformal import ConformalRegressor
from qms.optimization.buffer import DynamicBuffer
from qms.optimization.scheduler import POLICIES, get_policy
from qms.optimization.objectives import jain_fairness, composite_score
from qms.simulation.simulate import simulate


def get_or_prepare_real_data(cfg: dict) -> pd.DataFrame:
    real_csv = abspath(cfg["data"]["processed_dir"]) / "real_hospital_clean.csv"
    if not real_csv.exists():
        print("Real hospital clean CSV not found. Preparing now...")
        df_real = prepare_real_hospital(cfg)
    else:
        df_real = pd.read_csv(real_csv, parse_dates=["appointment_date", "check_in_time", "service_start_time", "service_end_time"], low_memory=False)
    return df_real


def evaluate_models(cfg: dict, df_synth: pd.DataFrame, df_real: pd.DataFrame) -> tuple[dict, dict]:
    tasks = ["no_show", "arrival_delay", "waiting_time", "service_time"]
    models_dir = abspath(cfg["models"]["dir"])
    real_metrics: dict = {}
    synth_metrics: dict = {}

    for task in tasks:
        bundle_path = models_dir / f"{task}.pkl"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle for {task} not found. Run scripts/02_train_models.py first.")
        bundle = ModelBundle.load(bundle_path)

        # 1. Synthetic test split evaluation
        X_s_raw, y_s, s_rows = preprocess.get_task_frame(df_synth, task)
        test_mask = preprocess.temporal_split(s_rows, cfg["data"]["test_size"])
        X_s_te = preprocess.transform(X_s_raw[test_mask], bundle.feature_columns, bundle.categorical).values
        y_s_te = y_s[test_mask].values

        if bundle.kind == "classification":
            proba_s = bundle.estimator.predict_proba(X_s_te)[:, 1]
            m_s = metrics.classification_metrics(y_s_te, proba_s, threshold=bundle.threshold)
        else:
            preds_s = bundle.estimator.predict(X_s_te)
            m_s = metrics.regression_metrics(y_s_te, preds_s)
        synth_metrics[task] = m_s

        # 2. Real hospital out-of-distribution evaluation
        X_r_raw, y_r, _ = preprocess.get_task_frame(df_real, task)
        X_r_te = preprocess.transform(X_r_raw, bundle.feature_columns, bundle.categorical).values
        y_r_te = y_r.values

        if bundle.kind == "classification":
            proba_r = bundle.estimator.predict_proba(X_r_te)[:, 1]
            m_r = metrics.classification_metrics(y_r_te, proba_r, threshold=bundle.threshold)
            m_r["predictions"] = proba_r
        else:
            preds_r = bundle.estimator.predict(X_r_te)
            m_r = metrics.regression_metrics(y_r_te, preds_r)
            m_r["predictions"] = preds_r

        m_r["n_test_samples"] = int(len(y_r_te))
        m_r["target_mean"] = round(float(np.mean(y_r_te)), 4) if len(y_r_te) else None
        real_metrics[task] = m_r

    return real_metrics, synth_metrics


def evaluate_conformal_on_real(df_synth: pd.DataFrame, df_real: pd.DataFrame, models_dir: Path) -> dict:
    results = {}
    for task in ["waiting_time", "service_time", "arrival_delay"]:
        bundle = ModelBundle.load(models_dir / f"{task}.pkl")

        # Fit calibrator on synthetic data
        X_s, y_s, _ = preprocess.get_task_frame(df_synth, task)
        X_s_enc = preprocess.transform(X_s, bundle.feature_columns, bundle.categorical).values
        preds_s = bundle.estimator.predict(X_s_enc)

        calibrator = ConformalRegressor(alpha=0.10, task=task)
        calibrator.calibrate(y_s.values, preds_s)

        # Evaluate intervals on real hospital dataset
        X_r, y_r, _ = preprocess.get_task_frame(df_real, task)
        X_r_enc = preprocess.transform(X_r, bundle.feature_columns, bundle.categorical).values
        preds_r = bundle.estimator.predict(X_r_enc)

        cov_dict = calibrator.evaluate_coverage(y_r.values, preds_r)
        emp_cov = float(cov_dict["empirical_coverage"])
        results[task] = {
            "target_confidence": 0.90,
            "empirical_coverage_on_real": round(emp_cov, 4),
            "mean_interval_width": round(float(cov_dict["mean_interval_width"]), 2),
            "calibration_quantile_qhat": round(float(calibrator.quantile_residual), 2),
            "valid_transfer": bool(emp_cov >= 0.90),
        }
    return results


def evaluate_buffer_on_real(df_real: pd.DataFrame, cfg: dict) -> dict:
    bundle = ModelBundle.load(abspath(cfg["models"]["dir"]) / "no_show.pkl")
    X_raw, _, rows = preprocess.get_task_frame(df_real, "no_show")
    rows = rows.copy()
    rows["noshow_prob"] = bundle.predict_proba(X_raw)

    buf = DynamicBuffer(
        base_minutes=cfg["buffer"]["base_minutes"],
        max_overbook_fraction=cfg["buffer"]["max_overbook_fraction"],
    )

    recs = []
    for _, g in rows.groupby(["appointment_date", "time_slot", "department"]):
        if len(g) < 2:
            continue
        recs.append(buf.recommend(g["noshow_prob"].values))

    if not recs:
        return {"available": False, "reason": "No multi-booking slots found in real dataset."}

    exp = np.array([r["expected_no_shows"] for r in recs])
    ob = np.array([r["recommended_overbook"] for r in recs])
    booked = np.array([r["n_booked"] for r in recs])

    return {
        "n_slots": int(len(recs)),
        "mean_bookings_per_slot": round(float(booked.mean()), 2),
        "mean_expected_no_shows_per_slot": round(float(exp.mean()), 3),
        "mean_recommended_overbook": round(float(ob.mean()), 3),
        "pct_slots_with_overbooking": round(float((ob > 0).mean() * 100), 1),
        "total_recommended_extra_bookings": int(ob.sum()),
        "capacity_uplift_pct": round(float(ob.sum() / booked.sum() * 100), 2),
    }


def evaluate_policies_on_real(df_real: pd.DataFrame, cfg: dict) -> dict:
    shown = df_real[(df_real["no_show"] == 0) & df_real["service_time"].notna() & df_real["check_in_time"].notna()].copy()
    shown["ready_min"] = (
        shown["check_in_time"].dt.hour * 60
        + shown["check_in_time"].dt.minute
        + shown["check_in_time"].dt.second / 60.0
    )

    doc_groups = []
    for (date_val, doc_id), g in shown.groupby(["appointment_date", "doctor_id"]):
        if len(g) < 2:
            continue
        patients = [
            {
                "id": int(r.appointment_id),
                "ready": float(r.ready_min),
                "service": float(r.service_time),
                "priority": int(r.priority_level),
            }
            for r in g.itertuples(index=False)
        ]
        doc_groups.append({"date": str(date_val), "doctor": str(doc_id), "n_servers": 1, "patients": patients})

    weights = cfg["scheduler"]["objective_weights"]
    emergency_level = cfg["scheduler"]["emergency_priority_level"]

    metric_rows: dict = {}
    for pname in POLICIES:
        policy = get_policy(pname)
        waits_all: list[float] = []
        priorities_all: list[int] = []
        total_busy = 0.0
        total_capacity = 0.0

        for group in doc_groups:
            res = simulate(group["patients"], 1, policy, emergency_level=emergency_level, weights=weights)
            waits_all.extend(res["waits"])
            priorities_all.extend(res["priorities"])
            total_busy += res["busy_time"]
            horizon = max(res["makespan"] - res["start"], 0.0)
            total_capacity += max(horizon, 0.001)

        waits = np.array(waits_all) if waits_all else np.array([0.0])
        prios = np.array(priorities_all) if priorities_all else np.array([0])
        emer_mask = prios >= emergency_level
        util = float(total_busy / total_capacity) if total_capacity > 0 else 0.0

        metric_rows[pname] = {
            "avg_wait": round(float(waits.mean()), 3),
            "p90_wait": round(float(np.percentile(waits, 90)), 3),
            "max_wait": round(float(waits.max()), 3),
            "utilization": round(min(util, 1.0), 4),
            "fairness": round(float(jain_fairness(waits)), 4),
            "emergency_wait": round(float(waits[emer_mask].mean()) if emer_mask.any() else 0.0, 3),
            "n_emergency": int(emer_mask.sum()),
            "n_patients": int(len(waits)),
        }

    scores = composite_score(metric_rows, weights)
    for p in metric_rows:
        metric_rows[p]["composite_score"] = scores[p]

    return metric_rows


def evaluate_fairness_on_real(df_real: pd.DataFrame) -> dict:
    shown = df_real[(df_real["no_show"] == 0) & df_real["waiting_time"].notna()].copy()
    median_wait = float(shown["waiting_time"].median())
    shown["timely_service"] = (shown["waiting_time"] <= median_wait).astype(int)

    # Across departments
    dept_rates = shown.groupby("department")["timely_service"].mean()
    min_rate = float(dept_rates.min())
    max_rate = float(dept_rates.max())
    dir_dept = round(float(min_rate / max_rate), 4) if max_rate > 0 else 1.0
    dpd_dept = round(float(max_rate - min_rate), 4)

    return {
        "sensitive_attribute": "department",
        "n_departments": int(len(dept_rates)),
        "disparate_impact_ratio": dir_dept,
        "demographic_parity_difference": dpd_dept,
        "four_fifths_rule_satisfied": bool(dir_dept >= 0.80),
        "department_timely_rates": {str(k): round(float(v), 4) for k, v in dept_rates.items()},
    }


def plot_real_results(real_metrics: dict, synth_metrics: dict, policies: dict, cfg: dict):
    figdir = abspath(cfg["reports"]["figures_dir"])
    figdir.mkdir(parents=True, exist_ok=True)

    # 1. Prediction comparison plot (Synthetic vs Real MAE/AUC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    reg_tasks = ["service_time", "waiting_time", "arrival_delay"]
    synth_maes = [synth_metrics[t]["mae"] for t in reg_tasks]
    real_maes = [real_metrics[t]["mae"] for t in reg_tasks]

    x = np.arange(len(reg_tasks))
    width = 0.35

    axes[0].bar(x - width/2, synth_maes, width, label="Synthetic Test", color="#4C72B0")
    axes[0].bar(x + width/2, real_maes, width, label="Real Hospital Test", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Service Time", "Waiting Time", "Arrival Delay"])
    axes[0].set_ylabel("MAE (minutes)")
    axes[0].set_title("Regression Error: Synthetic vs Real Transfer")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # No-show recall / precision
    clf_metrics = ["recall", "precision", "f1"]
    s_vals = [synth_metrics["no_show"][m] for m in clf_metrics]
    r_vals = [real_metrics["no_show"][m] for m in clf_metrics]

    x2 = np.arange(len(clf_metrics))
    axes[1].bar(x2 - width/2, s_vals, width, label="Synthetic Test", color="#4C72B0")
    axes[1].bar(x2 + width/2, r_vals, width, label="Real Hospital Test", color="#55A868")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["Recall", "Precision", "F1 Score"])
    axes[1].set_ylabel("Score")
    axes[1].set_title("No-Show Classification: Synthetic vs Real Transfer")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(figdir / "synthetic_vs_real_comparison.png", dpi=120)
    plt.close(fig)

    print(f"  [Saved Plot] -> {figdir / 'synthetic_vs_real_comparison.png'}")


def main():
    print("=" * 70)
    print("  STEP 5: OUT-OF-DISTRIBUTION REAL HOSPITAL EVALUATION")
    print("  (Trained ONLY on Synthetic Data -> Tested on Real Hospital Data)")
    print("=" * 70)

    cfg = load_config()
    ensure_dirs(cfg["reports"]["metrics_dir"], cfg["reports"]["figures_dir"])

    df_synth = load_clean(cfg)
    df_real = get_or_prepare_real_data(cfg)

    print(f"\n1. Datasets Loaded:")
    print(f"   * Synthetic Training Dataset: {len(df_synth):,} appointments")
    print(f"   * Real Hospital Test Dataset: {len(df_real):,} appointments")

    print("\n2. Evaluating Predictive Models Zero-Shot on Real Hospital...")
    real_metrics, synth_metrics = evaluate_models(cfg, df_synth, df_real)

    clean_real_metrics = {}
    for task, m in real_metrics.items():
        clean_m = {k: v for k, v in m.items() if k != "predictions"}
        clean_real_metrics[task] = clean_m
        if "mae" in clean_m:
            print(f"   [{task.upper():13s}] MAE: {clean_m['mae']:.2f} min | RMSE: {clean_m['rmse']:.2f} min | Skill: {clean_m['mae_skill']:.4f}")
        else:
            print(f"   [{task.upper():13s}] ROC-AUC: {clean_m['auc']:.4f} | Recall: {clean_m['recall']:.4f} | F1: {clean_m['f1']:.4f}")

    print("\n3. Evaluating Conformal Prediction Guarantees on Real Hospital...")
    models_dir = abspath(cfg["models"]["dir"])
    conformal_real = evaluate_conformal_on_real(df_synth, df_real, models_dir)
    for task, c in conformal_real.items():
        print(f"   [{task.upper():13s}] Target: {c['target_confidence']*100:.0f}% | Real Coverage: {c['empirical_coverage_on_real']*100:.2f}% | Interval Width: {c['mean_interval_width']:.1f} min")

    print("\n4. Evaluating Dynamic Buffer & Capacity Uplift on Real Hospital...")
    buffer_real = evaluate_buffer_on_real(df_real, cfg)
    print(f"   * Slots Analyzed: {buffer_real['n_slots']}")
    print(f"   * Safe Overbook per Slot: {buffer_real['mean_recommended_overbook']} patients")
    print(f"   * Real Hospital Capacity Uplift: +{buffer_real['capacity_uplift_pct']:.1f}%")

    print("\n5. Evaluating Scheduling Policies on Real Hospital Queues...")
    policies_real = evaluate_policies_on_real(df_real, cfg)
    for p, pm in policies_real.items():
        print(f"   [{p:15s}] Avg Wait: {pm['avg_wait']:.2f} min | Emerg Wait: {pm['emergency_wait']:.2f} min | Util: {pm['utilization']*100:.1f}% | Score: {pm['composite_score']:.3f}")

    print("\n6. Evaluating Algorithmic Fairness on Real Hospital Data...")
    fairness_real = evaluate_fairness_on_real(df_real)
    print(f"   * Disparate Impact Ratio: {fairness_real['disparate_impact_ratio']:.4f} (4/5ths Rule: {fairness_real['four_fifths_rule_satisfied']})")
    print(f"   * Demographic Parity Diff: {fairness_real['demographic_parity_difference']:.4f}")

    # Generate plots
    plot_real_results(real_metrics, synth_metrics, policies_real, cfg)

    # Save artifacts
    metrics_dir = abspath(cfg["reports"]["metrics_dir"])
    real_out = {
        "dataset": "realistic_hospital_testing_dataset_v8.csv",
        "n_samples": len(df_real),
        "protocol": "Trained exclusively on synthetic, tested out-of-distribution on real hospital",
        "models": clean_real_metrics,
        "conformal_prediction": conformal_real,
        "buffer_analysis": buffer_real,
        "policy_comparison": policies_real,
        "fairness_audit": fairness_real,
    }
    with open(metrics_dir / "real_hospital_test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(real_out, fh, indent=2)

    synth_vs_real = {
        "protocol": "Zero-shot transfer from synthetic clinic simulation to real-world hospital",
        "comparison": {
            task: {
                "synthetic_test": {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in synth_metrics[task].items() if k != "predictions"},
                "real_hospital_test": {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in clean_real_metrics[task].items()},
            }
            for task in ["no_show", "arrival_delay", "waiting_time", "service_time"]
        },
        "conformal_real_coverage": conformal_real,
        "buffer_real_capacity_uplift": buffer_real.get("capacity_uplift_pct"),
    }
    with open(metrics_dir / "synthetic_vs_real_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(synth_vs_real, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Real hospital test artifacts saved:")
    print(f"  * {metrics_dir / 'real_hospital_test_metrics.json'}")
    print(f"  * {metrics_dir / 'synthetic_vs_real_comparison.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
