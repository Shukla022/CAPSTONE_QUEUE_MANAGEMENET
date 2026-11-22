"""Evaluation metrics implemented in pure NumPy (no scikit-learn required)."""
from __future__ import annotations

from typing import Dict
import numpy as np


# ----------------------------- classification -----------------------------
def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney U) statistic."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true, scores, threshold: float = 0.5) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    c = confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_acc = 0.5 * (recall + specificity)
    youden_j = recall + specificity - 1.0
    base_rate = float(np.mean(np.asarray(y_true).astype(int)))
    flag_rate = float(np.mean(y_pred))
    return {
        "auc": round(roc_auc(y_true, scores), 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "youden_j": round(youden_j, 4),
        "base_rate": round(base_rate, 4),
        "flag_rate": round(flag_rate, 4),
        "threshold": threshold,
        **c,
    }


def choose_threshold(y_true, scores, *, min_recall: float = 0.20,
                     min_specificity: float = 0.50,
                     beta: float = 1.0) -> float:
    """Pick a decision threshold on a validation slice (not the test set).

    Balances F-beta and specificity so that the model flags genuinely high-risk
    patients without trivially labelling the entire clinic cohort as no-shows.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate([
        np.array([0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50]),
        np.quantile(scores, np.linspace(0.10, 0.90, 20)),
    ]))
    candidates = np.clip(candidates, 0.15, 0.70)

    def _fbeta(prec, rec):
        b2 = beta * beta
        return (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)

    best_t, best_score = 0.25, -1.0
    # Search for threshold satisfying both minimum recall and specificity
    for t in candidates:
        m = classification_metrics(y_true, scores, float(t))
        if m["recall"] < min_recall or m["specificity"] < min_specificity:
            continue
        s = _fbeta(m["precision"], m["recall"]) + 0.5 * m["balanced_accuracy"]
        if s > best_score:
            best_score, best_t = s, float(t)

    if best_score < 0:
        # Fallback to Youden's J index (balanced accuracy maximum)
        for t in candidates:
            m = classification_metrics(y_true, scores, float(t))
            s = m["youden_j"]
            if s > best_score:
                best_score, best_t = s, float(t)

    return round(float(best_t), 3)


def calibration_table(y_true, scores, n_bins: int = 10) -> list:
    """Reliability data: mean predicted vs observed frequency per probability bin."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        if m.sum() == 0:
            continue
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "mean_pred": round(float(scores[m].mean()), 4),
            "obs_freq": round(float(y_true[m].mean()), 4),
            "n": int(m.sum()),
        })
    return out


# ------------------------------- regression --------------------------------
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # For MAE, the mathematically optimal constant predictor is the median:
    baseline_median = float(np.mean(np.abs(y_true - np.median(y_true))))
    baseline_mean = float(np.mean(np.abs(y_true - y_true.mean())))
    
    skill_vs_median = 1.0 - mae / baseline_median if baseline_median > 0 else 0.0
    skill_vs_mean = 1.0 - mae / baseline_mean if baseline_mean > 0 else 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mean_target": round(float(y_true.mean()), 4),
        "median_target": round(float(np.median(y_true)), 4),
        "mae_baseline": round(baseline_median, 4),           # Honest median baseline for MAE
        "mae_baseline_median": round(baseline_median, 4),
        "mae_baseline_mean": round(baseline_mean, 4),
        "mae_skill": round(skill_vs_median, 4),              # Skill against optimal median constant
        "mae_skill_vs_median": round(skill_vs_median, 4),
        "mae_skill_vs_mean": round(skill_vs_mean, 4),
    }
