"""Train and evaluate all three predictive models, then persist artifacts.

Models
------
* no_show        -> classification  (target: no_show)          [booking-time features]
* arrival_delay  -> regression      (target: arrival_delay)    [booking-time features]
* waiting_time   -> regression      (target: waiting_time)     [check-in features]
* service_time   -> regression      (target: service_time)     [planning-time features]

Outputs
-------
* models/<task>.pkl                       -- portable ModelBundle
* reports/metrics/model_metrics.json      -- all test metrics + importances
* reports/figures/*.png                   -- importance / calibration / fit plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import load_config, abspath, ensure_dirs
from ..data.loader import load_clean
from ..data import preprocess
from . import estimators, metrics


def _fit_one(task: str, df, cfg):
    numeric, categorical, target, _ = preprocess.TASKS[task]
    X_raw, y, rows = preprocess.get_task_frame(df, task)
    test_mask = preprocess.temporal_split(rows, cfg["data"]["test_size"])

    train_rows = rows.loc[~test_mask]
    order_cols = ["appointment_date"]
    if "booking_timestamp" in train_rows.columns:
        order_cols.append("booking_timestamp")
    elif "appointment_id" in train_rows.columns:
        order_cols.append("appointment_id")
    train_order = train_rows.sort_values(order_cols).index

    Xtr_raw = X_raw.loc[train_order]
    ytr = y.loc[train_order].values
    Xte_raw, yte = X_raw[test_mask], y[test_mask].values

    feat_cols = preprocess.fit_encoder(Xtr_raw, categorical)
    Xtr = preprocess.transform(Xtr_raw, feat_cols, categorical).values
    Xte = preprocess.transform(Xte_raw, feat_cols, categorical).values

    n_val = max(int(round(0.15 * len(ytr))), 50)
    n_val = min(n_val, max(len(ytr) // 5, 1))
    X_fit, y_fit = Xtr[:-n_val], ytr[:-n_val]
    X_val, y_val = Xtr[-n_val:], ytr[-n_val:]

    if task == "no_show":
        kind = "classification"
        est = estimators.make_classifier(cfg["seed"])
        est.fit(X_fit, y_fit)
        val_proba = est.predict_proba(X_val)[:, 1]
        ns_cfg = cfg.get("models", {}).get("no_show", {})
        min_rec = float(ns_cfg.get("min_recall", 0.20))
        thr = metrics.choose_threshold(y_val, val_proba, min_recall=min_rec)
        
        # refit on full training set with unweighted honest probabilities
        est = estimators.make_classifier(cfg["seed"])
        est.fit(Xtr, ytr)
        proba = est.predict_proba(Xte)[:, 1]
        m = metrics.classification_metrics(yte, proba, threshold=thr)
        m["n_train"], m["n_test"] = int(len(ytr)), int(len(yte))
        m["threshold_source"] = "validation_balanced"
        calib = metrics.calibration_table(yte, proba)
        extra = {"calibration": calib}
        preds_for_plot = proba
    else:
        kind = "regression"
        thr = 0.5
        if task == "arrival_delay":
            loss = "absolute_error"
            extra_fit = estimators.arrival_regressor_params(cfg["seed"])
            extra_fit.pop("random_state", None)
            est = estimators.make_regressor(cfg["seed"], loss=loss, **extra_fit)
            est.fit(X_fit, y_fit)
            est = estimators.make_regressor(cfg["seed"], loss=loss, **extra_fit)
            est.fit(Xtr, ytr)
        else:
            loss = "squared_error"
            est = estimators.make_regressor(cfg["seed"], loss=loss)
            est.fit(X_fit, y_fit)
            est = estimators.make_regressor(cfg["seed"], loss=loss)
            est.fit(Xtr, ytr)
        preds = est.predict(Xte)
        m = metrics.regression_metrics(yte, preds)
        m["n_train"], m["n_test"] = int(len(ytr)), int(len(yte))
        extra = {}
        preds_for_plot = preds

    imp = estimators.top_importances(est, feat_cols, k=15)
    bundle = estimators.ModelBundle(
        task=task, kind=kind, target_name=target, estimator=est,
        feature_columns=feat_cols, numeric=numeric, categorical=categorical,
        backend=estimators.backend_name(), threshold=thr,
        metrics=m, feature_importances=imp,
    )
    bundle.save(abspath(cfg["models"]["dir"]) / f"{task}.pkl")

    _make_plots(task, kind, yte, preds_for_plot, imp, cfg, extra)
    return {"metrics": m, "feature_importances": imp, **extra}


def _make_plots(task, kind, y_true, preds, importances, cfg, extra):
    figdir = abspath(cfg["reports"]["figures_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    try:
        # Feature importance
        if importances:
            names = list(importances.keys())[::-1]
            vals = [importances[n] for n in names]
            plt.figure(figsize=(7, 5))
            plt.barh(names, vals, color="#4C72B0")
            plt.title(f"{task}: top feature importances")
            plt.tight_layout()
            plt.savefig(figdir / f"{task}_feature_importance.png", dpi=120)
            plt.close()

        if kind == "classification":
            calib = extra.get("calibration", [])
            if calib:
                mp = [c["mean_pred"] for c in calib]
                of = [c["obs_freq"] for c in calib]
                plt.figure(figsize=(5, 5))
                plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
                plt.plot(mp, of, "o-", color="#C44E52", label="model")
                plt.xlabel("mean predicted probability")
                plt.ylabel("observed no-show frequency")
                plt.title("No-show model calibration")
                plt.legend()
                plt.tight_layout()
                plt.savefig(figdir / f"{task}_calibration.png", dpi=120)
                plt.close()
        else:
            idx = np.random.default_rng(0).choice(
                len(y_true), size=min(2000, len(y_true)), replace=False)
            plt.figure(figsize=(5, 5))
            plt.scatter(np.asarray(y_true)[idx], np.asarray(preds)[idx],
                        s=6, alpha=0.3, color="#55A868")
            lim = [float(np.min(y_true)), float(np.max(y_true))]
            plt.plot(lim, lim, "--", color="gray")
            plt.xlabel("actual")
            plt.ylabel("predicted")
            plt.title(f"{task}: predicted vs actual")
            plt.tight_layout()
            plt.savefig(figdir / f"{task}_pred_vs_actual.png", dpi=120)
            plt.close()
    except Exception as exc:  # noqa: BLE001
        print(f"   [warn] plot failed for {task}: {exc}")


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs(cfg["models"]["dir"], cfg["reports"]["metrics_dir"],
                cfg["reports"]["figures_dir"])
    df = load_clean(cfg)

    print(f"Backend: {estimators.backend_name()}")
    results = {"backend": estimators.backend_name(), "models": {}}
    for task in ["no_show", "arrival_delay", "waiting_time", "service_time"]:
        print(f"\n>> Training '{task}' ...")
        res = _fit_one(task, df, cfg)
        results["models"][task] = res
        print(f"   metrics: {res['metrics']}")

    out = abspath(cfg["reports"]["metrics_dir"]) / "model_metrics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved metrics -> {out}")
    return results


if __name__ == "__main__":
    main()
