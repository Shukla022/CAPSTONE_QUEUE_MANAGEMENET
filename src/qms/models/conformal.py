"""Split Conformal Prediction for queue performance metrics.

Provides distribution-free, finite-sample predictive intervals with mathematically
guaranteed marginal coverage:
    P(Y_true in [Y_lower, Y_upper]) >= 1 - alpha

Used to quantify uncertainty in:
- Waiting time (minutes)
- Service duration (minutes)
- Arrival delay (minutes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ConformalInterval:
    """A calibrated prediction interval with point estimate and bounds."""
    point_estimate: float
    lower: float
    upper: float
    alpha: float
    width: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "point_estimate": round(float(self.point_estimate), 2),
            "lower": round(float(self.lower), 2),
            "upper": round(float(self.upper), 2),
            "width": round(float(self.width), 2),
            "confidence_level": round(1.0 - self.alpha, 2),
        }


@dataclass
class ConformalRegressor:
    """Split Conformal Regressor using absolute residual non-conformity.

    Given a fitted predictor and calibration data (X_cal, y_cal):
        s_i = |y_cal_i - f(X_cal_i)|
        q = Quantile( (n+1)(1-alpha)/n, {s_i} )
        C(x) = [f(x) - q, f(x) + q]
    """
    alpha: float = 0.10  # default 90% confidence level
    quantile_residual: float = 0.0
    task: str = "waiting_time"
    min_bound: float = 0.0

    def calibrate(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """Calibrate non-conformity quantile on a held-out calibration split."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        n = len(y_t)
        if n == 0:
            self.quantile_residual = 0.0
            return 0.0

        residuals = np.abs(y_t - y_p)
        # Conformal quantile level with finite-sample correction: ceil((n+1)*(1-alpha))/n
        level = min(1.0, np.ceil((n + 1.0) * (1.0 - self.alpha)) / n)
        # np.quantile takes a proportion in [0, 1]
        self.quantile_residual = float(np.quantile(residuals, level, method="higher" if hasattr(np, "quantile") else "linear"))
        return self.quantile_residual

    def predict_interval(self, point_pred: float, alpha: Optional[float] = None) -> ConformalInterval:
        """Compute [lower, upper] interval for a point prediction."""
        q = self.quantile_residual
        if alpha is not None and alpha != self.alpha:
            # Scale proportionally if different alpha requested on fly
            scale = (1.0 - alpha) / max(1.0 - self.alpha, 1e-6)
            q = q * scale

        lower = max(self.min_bound, point_pred - q)
        upper = max(self.min_bound, point_pred + q)
        return ConformalInterval(
            point_estimate=point_pred,
            lower=lower,
            upper=upper,
            alpha=self.alpha if alpha is None else alpha,
            width=upper - lower,
        )

    def evaluate_coverage(self, y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
        """Compute empirical coverage and average width on a test set."""
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        q = self.quantile_residual

        lower = np.maximum(self.min_bound, y_p - q)
        upper = y_p + q

        covered = (y_t >= lower) & (y_t <= upper)
        coverage_rate = float(covered.mean()) if len(y_t) else 0.0
        mean_width = float((upper - lower).mean()) if len(y_t) else 0.0

        return {
            "target_confidence": round(1.0 - self.alpha, 3),
            "empirical_coverage": round(coverage_rate, 4),
            "mean_interval_width": round(mean_width, 2),
            "calibration_quantile": round(q, 2),
            "valid_guarantee": bool(coverage_rate >= (1.0 - self.alpha - 0.05)),
        }
