# `models/`

Trained model bundles, written by `scripts/02_train_models.py`:

```
models/
├── no_show.pkl         # classification  (booking-time features)
├── arrival_delay.pkl   # regression      (booking-time features)
└── waiting_time.pkl    # regression      (check-in features)
```

These `.pkl` files are **generated** and git-ignored. Regenerate with:

```bash
python scripts/02_train_models.py
```

## What's in a bundle

Each file is a pickled `ModelBundle` (`src/qms/models/estimators.py`) containing
everything needed to score a raw feature row at inference time:

- the fitted `estimator`,
- the exact one-hot `feature_columns` learned at training,
- the `numeric` / `categorical` column lists,
- the `backend` used (`scikit-learn (GradientBoosting)` or `numpy-fallback`),
- the decision `threshold` (classifier), test `metrics`, and `feature_importances`.

Load and use one directly:

```python
from qms.models.estimators import ModelBundle
bundle = ModelBundle.load("models/no_show.pkl")
proba = bundle.predict_proba(raw_dataframe)   # booking-time features
```

## Backend

The factory prefers scikit-learn gradient-boosted trees. If scikit-learn isn't
installed, it falls back to the NumPy learners in `ml_lite.py` — same interface, so
bundles are portable either way. Install the ML extra to upgrade:
`pip install -e ".[ml]"`.
