"""The four scheduling objectives and how they are measured.

The system optimises, simultaneously:
  1. waiting      -> minimise average patient waiting time
  2. utilization  -> maximise doctor (server) utilisation
  3. fairness     -> keep waiting times equitable (Jain's index; no starvation)
  4. emergency    -> keep emergency-patient waiting times low

``summarize`` turns a raw simulation result into these metrics.
``composite_score`` combines *normalised* metrics across policies for ranking.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np


def jain_fairness(x: np.ndarray) -> float:
    """Jain's fairness index in [0,1]; 1.0 == perfectly equal.

    Applied to (waiting_time + 1) so that all-zero waits score 1.0.
    """
    x = np.asarray(x, dtype=float) + 1.0
    if len(x) == 0:
        return 1.0
    num = float(x.sum()) ** 2
    den = float(len(x) * np.sum(x ** 2))
    return num / den if den > 0 else 1.0


def summarize(sim: Dict, emergency_level: int = 1) -> Dict[str, float]:
    waits = np.asarray(sim["waits"], dtype=float)
    prio = np.asarray(sim["priorities"], dtype=float)
    busy = float(sim["busy_time"])
    n_servers = int(sim["n_servers"])
    span = float(sim["makespan"] - sim["start"])
    capacity = n_servers * span if span > 0 else 0.0
    util = busy / capacity if capacity > 0 else 0.0

    emer_mask = prio >= emergency_level
    emer_wait = float(waits[emer_mask].mean()) if emer_mask.any() else 0.0

    return {
        "avg_wait": round(float(waits.mean()) if len(waits) else 0.0, 3),
        "p90_wait": round(float(np.percentile(waits, 90)) if len(waits) else 0.0, 3),
        "max_wait": round(float(waits.max()) if len(waits) else 0.0, 3),
        "utilization": round(min(util, 1.0), 4),
        "fairness": round(jain_fairness(waits), 4),
        "emergency_wait": round(emer_wait, 3),
        "n_patients": int(len(waits)),
    }


def composite_score(metric_rows: Dict[str, Dict[str, float]],
                    weights: Dict[str, float]) -> Dict[str, float]:
    """Rank policies with a single normalised score in [0,1] (higher = better).

    ``metric_rows`` maps policy_name -> its summarize() dict. Each objective is
    min-max normalised across the compared policies so the score is relative.
    """
    policies = list(metric_rows)

    def col(key):
        return np.array([metric_rows[p][key] for p in policies], dtype=float)

    def norm(vals, higher_better):
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            return np.ones_like(vals)
        z = (vals - lo) / (hi - lo)
        return z if higher_better else 1.0 - z

    n_wait = norm(col("avg_wait"), higher_better=False)
    n_util = norm(col("utilization"), higher_better=True)
    n_fair = norm(col("fairness"), higher_better=True)
    n_emer = norm(col("emergency_wait"), higher_better=False)

    w = weights
    total = w["waiting"] + w["utilization"] + w["fairness"] + w["emergency"]
    score = (w["waiting"] * n_wait + w["utilization"] * n_util +
             w["fairness"] * n_fair + w["emergency"] * n_emer) / total
    return {p: round(float(s), 4) for p, s in zip(policies, score)}
