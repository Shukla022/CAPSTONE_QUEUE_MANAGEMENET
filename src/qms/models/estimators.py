"""Model factory + a portable inference bundle.

The factory prefers scikit-learn (gradient-boosted trees). If scikit-learn is
not importable it transparently falls back to the NumPy learners in
``ml_lite``. Either way, the rest of the codebase sees the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import numpy as np
import pandas as pd

from . import ml_lite
from ..data import preprocess

try:  # pragma: no cover - depends on environment
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    HAS_SKLEARN = True
except Exception:  # noqa: BLE001
    HAS_SKLEARN = False


def backend_name() -> str:
    return "scikit-learn (GradientBoosting)" if HAS_SKLEARN else "numpy-fallback"


def make_classifier(seed: int = 42, **kwargs):
    if HAS_SKLEARN:
        params = dict(n_estimators=180, max_depth=3, learning_rate=0.05,
                      subsample=0.85, min_samples_leaf=25, random_state=seed)
        params.update(kwargs)
        return GradientBoostingClassifier(**params)
    return ml_lite.LogisticRegressionLite(seed=seed)


def make_regressor(seed: int = 42, *, loss: str = "squared_error", **kwargs):
    if HAS_SKLEARN:
        params = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                      subsample=0.85, min_samples_leaf=20, random_state=seed)
        params.update(kwargs)
        # Huber is more robust to late/early outliers (arrival delay).
        try:
            return GradientBoostingRegressor(loss=loss, **params)
        except TypeError:
            params.pop("loss", None)
            return GradientBoostingRegressor(**params)
    return ml_lite.RidgeRegressionLite(l2=1.0)


def arrival_regressor_params(seed: int = 42) -> dict:
    """Robust L1/absolute-error regressor for punctuality, directly targeting MAE."""
    return dict(
        n_estimators=120, max_depth=3, learning_rate=0.04,
        subsample=0.85, min_samples_leaf=40,
        random_state=seed,
    )


def balanced_sample_weights(y) -> np.ndarray:
    """Inverse-frequency weights so the minority class is not ignored."""
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    w = np.ones(n, dtype=float)
    w[y == 1] = n / (2.0 * n_pos)
    w[y == 0] = n / (2.0 * n_neg)
    return w


@dataclass
class ModelBundle:
    """Everything needed to score a raw feature row at inference time."""

    task: str
    kind: str                      # "classification" | "regression"
    target_name: str
    estimator: object
    feature_columns: List[str]
    numeric: List[str]
    categorical: List[str]
    backend: str
    threshold: float = 0.5
    metrics: Dict = field(default_factory=dict)
    feature_importances: Dict[str, float] = field(default_factory=dict)

    # -- inference -----------------------------------------------------
    def _design_matrix(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = preprocess.transform(X_raw, self.feature_columns, self.categorical)
        return X.values

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        if self.kind != "classification":
            raise ValueError("predict_proba is only valid for classifiers")
        return self.estimator.predict_proba(self._design_matrix(X_raw))[:, 1]

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        if self.kind == "classification":
            return (self.predict_proba(X_raw) >= self.threshold).astype(int)
        return np.asarray(self.estimator.predict(self._design_matrix(X_raw)), dtype=float)

    # -- persistence ---------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        with open(Path(path), "rb") as fh:
            try:
                return pickle.load(fh)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Could not load '{path}'. Retrain in this environment: "
                    "python scripts/02_train_models.py"
                ) from exc


def top_importances(estimator, feature_columns: List[str], k: int = 15) -> Dict[str, float]:
    imp = getattr(estimator, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp, dtype=float)
    order = np.argsort(imp)[::-1][:k]
    return {feature_columns[i]: round(float(imp[i]), 4) for i in order}
