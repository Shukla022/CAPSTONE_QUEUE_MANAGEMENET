# Panel Speaker Notes — Model-Training Module

**Project:** Multi-Objective AI-Driven Cloud-Based Queue Management System
**My part:** Predictive model training pipeline
**Source of every number below:** `reports/metrics/model_metrics.json` (the committed training run)

---

## 30-second opening (say this first)

> "My part is the predictive layer. We train **four models** that feed the
> scheduler and the overbooking buffer. Everything runs from one command —
> `python scripts/02_train_models.py` — which loops over the four tasks, does a
> **time-based train/test split** so we never leak the future into the past,
> fits each model, computes test-set metrics, and saves each one as a portable
> bundle. Let me walk through the files, then each model and its accuracy."

---

## The pipeline (file flow — draw this on the board if you can)

```
data/raw/hospital_dataset_preprocessed.csv
   │
   ├─ scripts/01_prepare_data.py   → clean + profile → data/processed/clean.csv
   │
   └─ scripts/02_train_models.py   → qms.models.train_all.main(cfg)
          │
          ├─ src/qms/data/preprocess.py    feature engineering + leakage-safe split
          ├─ src/qms/models/estimators.py  model factory + ModelBundle (save/load)
          ├─ src/qms/models/metrics.py     pure-NumPy metrics (AUC, MAE, R², calibration)
          └─ src/qms/models/ml_lite.py     dependency-free fallback learners
          │
          └─ outputs:
               models/<task>.pkl                    4 portable model bundles
               reports/metrics/model_metrics.json   all metrics + feature importances
               reports/figures/*.png                importance / calibration / fit plots
```

**What `train_all.py` does per task** (`_fit_one`): get leakage-safe feature frame →
temporal split → one-hot encode & align columns → fit on training rows →
(no-show only: tune the decision threshold on a validation slice) → refit on full
train → compute test metrics → save a `ModelBundle` `.pkl` → make plots → dump JSON.

---

## The four models — what / why / number / what it feeds

### 1. No-show — classification (`GradientBoostingClassifier`)
- **Predicts:** probability a booked patient won't show. **Booking-time features only.**
- **Feeds:** the no-show-aware overbooking buffer.
- **Numbers:** AUC **0.596**, recall **99.62%**, precision **21.1%**, accuracy **25.6%**,
  threshold **0.15** (base no-show rate 19.9%).
- **Say:** "We optimise for **recall**, not accuracy — catching a no-show lets us
  reclaim the slot, while a false alarm only costs a small buffer. The model is
  **well-calibrated**, and it's the calibrated probability the buffer consumes."

### 2. Service time — regression (`GradientBoostingRegressor`) ⭐ strongest
- **Predicts:** consult duration from department, weekday, time-of-day. Planning-time features only.
- **Feeds:** slot-width sizing in the scheduler (the key ML input for scheduling).
- **Numbers:** R² **0.834**, MAE **2.10 min**, **+66.2% skill** over the mean baseline.
- **Say:** "This is our best model. Department is the dominant driver, which is
  clinically sensible — a cardiology consult isn't a general-medicine consult."

### 3. Waiting time — regression
- **Predicts:** real-time wait, produced **at check-in** — so it may use
  `queue_length_at_arrival` (only known once the patient is present).
- **Feeds:** the live estimate shown to patients/staff.
- **Numbers:** R² **0.134**, MAE **8.67 min**, **+6.4% skill** over a dept-average baseline.
- **Say:** "Modest but real skill. Top driver is queue-per-doctor."

### 4. Arrival delay — regression (deliberately regularised)
- **Predicts:** how early/late a patient arrives. Trained on shown patients only.
- **Feeds:** buffer window / sequencing.
- **Numbers:** MAE **8.47 min**, R² **≈ 0** (at baseline).
- **Say:** "Punctuality is close to random here, so we regularise hard —
  depth-2 trees, Huber loss, winsorised tails (`arrival_regressor_params`) — to
  avoid overfitting noise. Sitting at baseline is an honest finding, not a bug."

---

## Cheat-sheet table (memorise this)

| Model | Type | Headline metric | Baseline | Skill | Drives |
|---|---|---|---|---|---|
| No-show | Classifier | Recall **99.62%** (AUC 0.596) @ thr 0.15 | 19.9% base rate | — | Overbooking buffer |
| Service time | Regressor | **R² 0.834**, MAE 2.10 min | MAE 6.19 | **+66.2%** | Slot-width sizing |
| Waiting time | Regressor | R² 0.134, MAE 8.67 min | MAE 9.26 | +6.4% | Live check-in estimate |
| Arrival delay | Regressor | MAE 8.47 min, R² ≈ 0 | MAE 8.41 | ~0% | Buffer / sequencing |

---

## Why Gradient Boosting? (the "why this model" answer)

1. **Tabular, mixed data** (numeric + one-hot categorical) — tree ensembles are the
   standard choice and need no feature scaling.
2. **Non-linear interactions** captured automatically (e.g. priority × lead-time).
3. **Interpretable** — `feature_importances_` is exactly what the importance plots show.
4. **Robust to outliers**, especially with Huber loss on the noisy arrival-delay target.

**Engineering point:** the factory in `estimators.py` is backend-agnostic — it uses
scikit-learn Gradient Boosting when installed and a dependency-free **NumPy fallback**
(`ml_lite.py`: `LogisticRegressionLite`, `RidgeRegressionLite`) otherwise, behind one
identical interface. The run recorded in `model_metrics.json` names the backend used,
and the project runs anywhere with no code change.

---

## Two methodology points panels reward

**Leakage-safe temporal split** (`preprocess.py`): train on the earliest 80% of
appointments, test on the latest 20% — mimics deployment. Feature sets are gated by
*when the info is known*: no-show/arrival get booking-time features only; waiting-time
adds `queue_length_at_arrival`; history features use a lagged expanding mean
(`shift(1)`) so this visit's own outcome never leaks in.

**Honest threshold selection** (`metrics.choose_threshold`): the no-show cut-off is
picked on a **validation slice** (last 15% of training data), never the test set —
maximise F-β subject to a minimum recall. That's why it's 0.15, not a naive 0.5.

---

## Likely questions + crisp answers

**Q: 25% accuracy is bad, isn't it?**
A: Accuracy is the wrong metric for a rare event with asymmetric costs. "Predict
everyone shows" scores 80% accuracy but catches zero no-shows. We tune for recall,
catch 99.6% of no-shows, and the calibrated probability — not the hard label — is what
the buffer uses. AUC 0.60 measures ranking ability.

**Q: Why is arrival-delay R² about zero?**
A: Because patient punctuality is nearly random in this dataset. Rather than fake
skill by overfitting, we regularise heavily and report it honestly. It still supplies
a reasonable average for the buffer window.

**Q: Did you avoid data leakage?**
A: Yes — temporal split plus per-task feature gating by information-availability time,
and lagged (`shift(1)`) history features. Details are documented in the `preprocess.py` docstring.

**Q: Where do the numbers come from / can you reproduce them?**
A: `reports/metrics/model_metrics.json`, produced by `scripts/02_train_models.py`.
Re-running the script regenerates the models and the metrics file.

**Q: Why not deep learning?**
A: ~20k tabular rows — gradient-boosted trees outperform neural nets in this regime,
train in seconds, and are interpretable via feature importances.
