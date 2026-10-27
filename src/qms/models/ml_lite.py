"""Dependency-free NumPy learners used as a fallback when scikit-learn is absent.

These are intentionally simple (regularised logistic / ridge regression with
internal standardisation). When scikit-learn IS installed, the factory in
``estimators.py`` uses gradient-boosted trees instead, which are typically more
accurate. Keeping this fallback means the whole pipeline runs with only
pandas + numpy.
"""
from __future__ import annotations

import numpy as np


class _Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class LogisticRegressionLite:
    """L2-regularised logistic regression trained with gradient descent."""

    def __init__(self, lr: float = 0.1, n_iter: int = 800, l2: float = 1e-3, seed: int = 42):
        self.lr, self.n_iter, self.l2, self.seed = lr, n_iter, l2, seed

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.scaler_ = _Standardizer().fit(X)
        Xs = self.scaler_.transform(X)
        n, d = Xs.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(Xs @ self.w_ + self.b_)
            grad_w = Xs.T @ (p - y) / n + self.l2 * self.w_
            grad_b = float(np.mean(p - y))
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        p = self._sigmoid(Xs @ self.w_ + self.b_)
        return np.column_stack([1 - p, p])

    @property
    def feature_importances_(self):
        w = np.abs(self.w_)
        s = w.sum()
        return w / s if s > 0 else w


class RidgeRegressionLite:
    """Closed-form ridge regression with internal standardisation."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.scaler_ = _Standardizer().fit(X)
        Xs = self.scaler_.transform(X)
        n, d = Xs.shape
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_
        A = Xs.T @ Xs + self.l2 * np.eye(d)
        self.w_ = np.linalg.solve(A, Xs.T @ yc)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return Xs @ self.w_ + self.y_mean_

    @property
    def feature_importances_(self):
        w = np.abs(self.w_)
        s = w.sum()
        return w / s if s > 0 else w
