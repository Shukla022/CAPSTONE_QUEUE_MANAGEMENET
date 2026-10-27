"""Tests for the 4 research paper novelties:
1. Conformal Prediction uncertainty intervals
2. Chance-Constrained Overbooking Buffer
3. Pareto Frontier Analysis
4. Counterfactual GenAI Explanations & Algorithmic Fairness Audit
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

from qms.models.conformal import ConformalRegressor
from qms.optimization.buffer import DynamicBuffer
from qms.optimization.pareto import ParetoFrontierAnalyzer, evaluate_schedule_permutation
from qms.genai.counterfactual import CounterfactualEngine
from qms.genai.explainer import GenAIExplainer
from qms.evaluation.fairness_audit import audit_fairness_classification, audit_fairness_waiting_time


def test_conformal_prediction():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(10, 30, 200)
    y_pred = y_true + rng.normal(0, 3, 200)

    conf = ConformalRegressor(alpha=0.10)
    q = conf.calibrate(y_true[:100], y_pred[:100])
    assert q > 0

    interval = conf.predict_interval(20.0)
    assert interval.lower <= 20.0 <= interval.upper
    assert interval.width == (interval.upper - interval.lower)

    metrics = conf.evaluate_coverage(y_true[100:], y_pred[100:])
    assert metrics["empirical_coverage"] >= 0.80  # valid finite sample coverage
    assert metrics["valid_guarantee"] is True


def test_chance_constrained_buffer():
    buf = DynamicBuffer(base_minutes=10.0, max_overbook_fraction=0.30)
    noshow_probs = [0.2, 0.2, 0.3, 0.1, 0.25, 0.3, 0.15, 0.2, 0.1, 0.2]
    rec = buf.chance_constrained_recommend(noshow_probs, slot_capacity=10, alpha_risk=0.05)

    assert "chance_constrained_overbook" in rec
    assert rec["chance_constrained_overbook"] <= rec["max_overbook_allowed"]
    assert rec["tail_risk_respected"] is True

    # Test conformal buffer pad
    cbuf = buf.conformal_slot_buffer(
        wait_upper_bound=15.0,
        service_upper_bound=18.0,
        planned_service=12.0,
        noshow_probs=noshow_probs,
    )
    assert cbuf["total_conformal_buffer"] > cbuf["effective_buffer_minutes"]

    # Crowd pressure must shrink the gap vs an empty waiting room
    quiet = buf.slot_buffer(predicted_wait=10.0, noshow_probs=noshow_probs,
                            waiting_count=0, pending_count=0)
    crowded = buf.slot_buffer(predicted_wait=10.0, noshow_probs=noshow_probs,
                              waiting_count=8, pending_count=12)
    assert crowded < quiet
    assert crowded >= 1.0  # floor from min_minutes default


def test_pareto_frontier_analysis():
    patients = [
        {"id": 1, "ready": 0.0, "service": 15.0, "priority": 0},
        {"id": 2, "ready": 5.0, "service": 10.0, "priority": 1},
        {"id": 3, "ready": 10.0, "service": 8.0, "priority": 0},
        {"id": 4, "ready": 12.0, "service": 20.0, "priority": 1},
    ]
    analyzer = ParetoFrontierAnalyzer(emergency_level=1, max_permutations=24)
    res = analyzer.analyze_cohort(patients, n_servers=1)

    assert res["n_patients"] == 4
    assert res["n_pareto_optimal_points"] >= 1
    assert 0.0 <= res["pareto_efficiency_ratio"] <= 1.0


def test_counterfactual_engine_and_explainer():
    engine = CounterfactualEngine()
    cf = engine.what_if_arrival_punctuality(actual_wait=20.0, actual_delay=15.0)
    assert cf.counterfactual_value < cf.factual_value
    assert cf.delta_minutes > 0

    explainer = GenAIExplainer()
    text = explainer.explain_counterfactual("arrival_punctuality", cf.to_dict())
    assert len(text) > 20
    assert "wait" in text.lower()


def test_fairness_audit():
    df = pd.DataFrame({
        "department": ["Cardiology"] * 50 + ["Dermatology"] * 50,
        "no_show": [0, 1] * 50,
        "pred_no_show": [0, 1] * 50,
        "waiting_time": [10.0] * 50 + [12.0] * 50,
        "priority_level": [0] * 80 + [1] * 20,
    })
    clf_res = audit_fairness_classification(df, "department")
    assert clf_res["disparate_impact_ratio"] >= 0.80
    assert clf_res["four_fifths_rule_satisfied"] is True

    wait_res = audit_fairness_waiting_time(df, "department")
    assert wait_res["jains_fairness_index"] >= 0.90


if __name__ == "__main__":
    test_conformal_prediction()
    print("PASS: test_conformal_prediction")
    test_chance_constrained_buffer()
    print("PASS: test_chance_constrained_buffer")
    test_pareto_frontier_analysis()
    print("PASS: test_pareto_frontier_analysis")
    test_counterfactual_engine_and_explainer()
    print("PASS: test_counterfactual_engine_and_explainer")
    test_fairness_audit()
    print("PASS: test_fairness_audit")
    print("All novelty tests passed successfully!")
