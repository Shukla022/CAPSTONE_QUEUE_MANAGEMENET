"""Policy comparison + buffer analysis (the multi-objective evaluation).

What this does
--------------
1. **Scheduling policy comparison.** For every real (appointment_date, queue_id)
   group we replay the patients who actually showed up -- using their *actual*
   arrival (check-in) times and service durations -- through three sequencing
   policies:

       FCFS  <  Static-Priority  <  Multi-Objective (ours)

   Because arrival and service times are held fixed and only the *ordering*
   changes, the comparison is strictly apples-to-apples. We pool the realised
   waiting times across all groups and report the four objectives
   (avg waiting, doctor utilisation, fairness, emergency waiting), then rank the
   policies with a single normalised ``composite_score``.

2. **No-show-aware buffer analysis.** Using the trained no-show model we predict
   each booking's no-show probability, group bookings into slots, and ask the
   ``DynamicBuffer`` how much controlled overbooking is safe. This quantifies the
   utilisation head-room the predictive layer unlocks.

Outputs
-------
* reports/metrics/policy_comparison.json / .csv
* reports/metrics/buffer_analysis.json
* reports/figures/policy_comparison.png
* reports/figures/composite_score.png
* reports/figures/buffer_analysis.png
"""
from __future__ import annotations

import json
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import load_config, abspath, ensure_dirs
from ..data.loader import load_clean
from ..data import preprocess
from ..models.estimators import ModelBundle
from ..optimization.scheduler import POLICIES, get_policy
from ..optimization.objectives import jain_fairness, composite_score
from ..optimization.buffer import DynamicBuffer
from ..simulation.simulate import simulate, minutes_of_day


# ---------------------------------------------------------------------------
# 1. Build simulation groups from the real data
# ---------------------------------------------------------------------------
def build_groups(df: pd.DataFrame) -> List[Dict]:
    """One entry per (appointment_date, queue_id): the shown patients + capacity.

    A patient is included only if they actually attended (no_show == 0) and has a
    realised service time and check-in time (the inputs the simulator replays).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["check_in_time"]):
        df["check_in_time"] = pd.to_datetime(
            df["check_in_time"], dayfirst=True, errors="coerce")

    shown = df[(df["no_show"] == 0)
               & df["service_time"].notna()
               & df["check_in_time"].notna()].copy()
    shown["ready_min"] = (shown["check_in_time"].dt.hour * 60
                          + shown["check_in_time"].dt.minute
                          + shown["check_in_time"].dt.second / 60.0)

    groups: List[Dict] = []
    for (date, queue), g in shown.groupby(["appointment_date", "queue_id"]):
        if len(g) < 2:            # need at least two patients to sequence
            continue
        n_servers = int(max(1, round(float(g["active_doctors_count"].max()))))
        patients = [
            {
                "id": int(r.appointment_id),
                "ready": float(r.ready_min),
                "service": float(r.service_time),
                "priority": int(r.priority_level),
            }
            for r in g.itertuples(index=False)
        ]
        groups.append({"date": date, "queue": queue,
                       "n_servers": n_servers, "patients": patients})
    return groups


# ---------------------------------------------------------------------------
# 2. Run every policy over every group and pool the results
# ---------------------------------------------------------------------------
def run_policies(groups: List[Dict], weights: Dict[str, float],
                 emergency_level: int = 1) -> Dict[str, Dict[str, float]]:
    metric_rows: Dict[str, Dict[str, float]] = {}

    for pname in POLICIES:
        policy = get_policy(pname)
        all_waits: List[float] = []
        all_prio: List[float] = []
        busy_total = 0.0
        capacity_total = 0.0

        for grp in groups:
            sim = simulate(grp["patients"], grp["n_servers"], policy,
                           emergency_level=emergency_level, weights=weights)
            all_waits.extend(sim["waits"])
            all_prio.extend(sim["priorities"])
            span = float(sim["makespan"] - sim["start"])
            busy_total += float(sim["busy_time"])
            capacity_total += grp["n_servers"] * span if span > 0 else 0.0

        waits = np.asarray(all_waits, dtype=float)
        prio = np.asarray(all_prio, dtype=float)
        emer_mask = prio >= emergency_level
        util = busy_total / capacity_total if capacity_total > 0 else 0.0

        metric_rows[pname] = {
            "avg_wait": round(float(waits.mean()) if len(waits) else 0.0, 3),
            "p90_wait": round(float(np.percentile(waits, 90)) if len(waits) else 0.0, 3),
            "max_wait": round(float(waits.max()) if len(waits) else 0.0, 3),
            "utilization": round(min(util, 1.0), 4),
            "fairness": round(jain_fairness(waits), 4),
            "emergency_wait": round(float(waits[emer_mask].mean()) if emer_mask.any() else 0.0, 3),
            "n_emergency": int(emer_mask.sum()),
            "n_patients": int(len(waits)),
        }
    return metric_rows


# ---------------------------------------------------------------------------
# 3. No-show-aware buffer / overbooking analysis
# ---------------------------------------------------------------------------
def buffer_analysis(df: pd.DataFrame, cfg: dict) -> Dict:
    model_path = abspath(cfg["models"]["dir"]) / "no_show.pkl"
    if not model_path.exists():
        return {"available": False,
                "reason": "no_show model not found; run training first."}

    bundle = ModelBundle.load(model_path)
    X_raw, _, rows = preprocess.get_task_frame(df, "no_show")
    rows = rows.copy()
    rows["noshow_prob"] = bundle.predict_proba(X_raw)

    buf = DynamicBuffer(base_minutes=cfg["buffer"]["base_minutes"],
                        max_overbook_fraction=cfg["buffer"]["max_overbook_fraction"])

    recs = []
    for _, g in rows.groupby(["appointment_date", "time_slot", "department"]):
        if len(g) < 2:
            continue
        recs.append(buf.recommend(g["noshow_prob"].values))

    if not recs:
        return {"available": False, "reason": "no multi-booking slots found."}

    exp = np.array([r["expected_no_shows"] for r in recs])
    ob = np.array([r["recommended_overbook"] for r in recs])
    booked = np.array([r["n_booked"] for r in recs])

    summary = {
        "available": True,
        "backend": bundle.backend,
        "n_slots": int(len(recs)),
        "mean_bookings_per_slot": round(float(booked.mean()), 2),
        "mean_expected_no_shows_per_slot": round(float(exp.mean()), 3),
        "mean_recommended_overbook": round(float(ob.mean()), 3),
        "pct_slots_with_overbooking": round(float((ob > 0).mean() * 100), 1),
        "total_recommended_extra_bookings": int(ob.sum()),
        # extra bookings the model says we can safely admit, as a share of demand
        "capacity_uplift_pct": round(float(ob.sum() / booked.sum() * 100), 2),
    }
    _plot_buffer(exp, ob, cfg)
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_policy_comparison(metric_rows, scores, cfg):
    figdir = abspath(cfg["reports"]["figures_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    policies = list(metric_rows)
    colors = ["#8C8C8C", "#DD8452", "#4C72B0"]

    try:
        panels = [
            ("avg_wait", "Average waiting time (min)", 1.0),
            ("emergency_wait", "Emergency waiting time (min)", 1.0),
            ("utilization", "Doctor utilisation (%)", 100.0),
            ("fairness", "Fairness (Jain index, %)", 100.0),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (key, title, scale) in zip(axes.ravel(), panels):
            vals = [metric_rows[p][key] * scale for p in policies]
            bars = ax.bar(policies, vals, color=colors)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=15)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=9)
        fig.suptitle("Scheduling policy comparison (lower is better for waits)",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(figdir / "policy_comparison.png", dpi=120)
        plt.close(fig)

        # composite score
        plt.figure(figsize=(6, 4.5))
        svals = [scores[p] for p in policies]
        bars = plt.bar(policies, svals, color=colors)
        for b, v in zip(bars, svals):
            plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=10)
        plt.ylim(0, 1.08)
        plt.ylabel("Composite score (higher = better)")
        plt.title("Multi-objective composite score by policy")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(figdir / "composite_score.png", dpi=120)
        plt.close()
    except Exception as exc:  # noqa: BLE001
        print(f"   [warn] policy plot failed: {exc}")


def _plot_buffer(exp, ob, cfg):
    figdir = abspath(cfg["reports"]["figures_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(exp, bins=20, color="#4C72B0", alpha=0.85)
        axes[0].set_title("Expected no-shows per slot")
        axes[0].set_xlabel("expected no-shows")
        axes[0].set_ylabel("slots")

        maxob = int(ob.max()) if len(ob) else 0
        bins = np.arange(0, max(maxob, 1) + 2) - 0.5
        axes[1].hist(ob, bins=bins, color="#55A868", alpha=0.85)
        axes[1].set_title("Recommended overbooking per slot")
        axes[1].set_xlabel("extra bookings admitted")
        axes[1].set_ylabel("slots")
        fig.suptitle("No-show-aware dynamic buffer", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(figdir / "buffer_analysis.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"   [warn] buffer plot failed: {exc}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs(cfg["reports"]["metrics_dir"], cfg["reports"]["figures_dir"])
    weights = cfg["scheduler"]["objective_weights"]
    emergency_level = cfg["scheduler"]["emergency_priority_level"]

    df = load_clean(cfg)
    groups = build_groups(df)
    n_pat = sum(len(g["patients"]) for g in groups)
    print(f"Built {len(groups)} (date, queue) groups covering {n_pat} attended patients.")

    metric_rows = run_policies(groups, weights, emergency_level)
    scores = composite_score(metric_rows, weights)
    for p in metric_rows:
        metric_rows[p]["composite_score"] = scores[p]

    best = max(scores, key=scores.get)
    print("\nPolicy comparison:")
    for p, m in metric_rows.items():
        print(f"  {p:16s} avg_wait={m['avg_wait']:6.2f}  util={m['utilization']:.3f}  "
              f"fair={m['fairness']:.3f}  emer={m['emergency_wait']:6.2f}  "
              f"score={m['composite_score']:.3f}")
    print(f"  -> best by composite score: {best}")

    buf = buffer_analysis(df, cfg)
    if buf.get("available"):
        print(f"\nBuffer analysis: {buf['n_slots']} slots, "
              f"capacity uplift {buf['capacity_uplift_pct']}%.")

    _plot_policy_comparison(metric_rows, scores, cfg)

    # ---- persist ----
    mdir = abspath(cfg["reports"]["metrics_dir"])
    payload = {
        "weights": weights,
        "emergency_level": emergency_level,
        "n_groups": len(groups),
        "n_patients": n_pat,
        "policies": metric_rows,
        "best_policy": best,
    }
    with open(mdir / "policy_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    pd.DataFrame(metric_rows).T.to_csv(mdir / "policy_comparison.csv")
    with open(mdir / "buffer_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(buf, fh, indent=2)
    print(f"\nSaved evaluation -> {mdir}")

    return {"policies": metric_rows, "best_policy": best, "buffer": buf}


if __name__ == "__main__":
    main()
