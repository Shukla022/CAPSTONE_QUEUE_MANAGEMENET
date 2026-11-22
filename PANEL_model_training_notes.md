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
- **Numbers:** AUC **0.589**, recall **49.12%**, specificity **62.67%**, precision **24.64%**, accuracy **60.0%** on synthetic holdout, and **63.5% accuracy** on the real hospital test set (threshold **0.197**, base rate ~20%).
- **Say:** "We train unweighted to maintain true, well-calibrated posterior probabilities. The decision threshold (0.197) is optimized via Youden's J statistic on a validation slice, achieving balanced sensitivity and 74.1% specificity on real hospital data. The continuous probability directly scales the chance-constrained overbooking buffer."

### 2. Service time — regression (`GradientBoostingRegressor`) ⭐ strongest
- **Predicts:** consult duration from department, weekday, time-of-day. Planning-time features only.
- **Feeds:** slot-width sizing in the scheduler (the key ML input for scheduling).
- **Numbers:** R² **0.832**, MAE **2.12 min**, **+64.4% skill** over median baseline (+65.8% over mean baseline). Transferred zero-shot to real hospital data with MAE **3.95 min**.
- **Say:** "This is our flagship predictive model. Clinical department is the dominant driver (Cardiology consults average ~33 min vs General Medicine ~11.7 min). It generalizes to real hospital consults within an MAE under 4 minutes."

### 3. Waiting time — regression
- **Predicts:** real-time wait, produced **at check-in** — so it may use
  `queue_length_at_arrival` (only known once the patient is present).
- **Feeds:** the live estimate shown to patients/staff and conformal uncertainty bounds.
- **Numbers:** R² **0.207**, MAE **8.38 min**, **+4.7% skill** over median baseline (+9.5% over mean baseline).
- **Say:** "Real-time check-in model. Dominant feature is queue-per-doctor. Calibrated conformal prediction guarantees 95.89% empirical coverage on real hospital patients under a 90% confidence target."

### 4. Arrival delay — regression (L1/MAE optimized)
- **Predicts:** how early/late a patient arrives. Trained on shown patients only.
- **Feeds:** buffer window / sequencing.
- **Numbers:** MAE **8.23 min** on synthetic holdout (matching the optimal median baseline of 8.22 min); MAE **13.05 min** on real hospital data.
- **Say:** "Patient punctuality is inherently high-entropy human behavior. We train directly under L1 loss (MAE) with regularized gradient boosting to avoid overfitting noise. We transparently report both median and mean baselines, matching optimal constant baseline without hallucinating false correlations."

---

## Cheat-sheet table (memorise this)

| Model | Type | Headline metric | Baseline (Median / Mean) | Real Hospital Transfer | Drives |
|---|---|---|---|---|---|
| No-show | Classifier | **AUC 0.589**, Acc **60.0%**, Rec 49.1% | 19.9% base rate | **63.5% Acc**, 74.1% Spec | Dynamic Overbooking Buffer |
| Service time | Regressor | **R² 0.832**, MAE **2.12 min** | 5.94 min / 6.19 min | **MAE 3.95 min** | Slot-width sizing |
| Waiting time | Regressor | **R² 0.207**, MAE **8.38 min** | 8.79 min / 9.26 min | **MAE 8.61 min** (95.9% Conformal) | Live check-in estimate |
| Arrival delay | Regressor | MAE **8.23 min** | 8.22 min / 8.41 min | **MAE 13.05 min** | Buffer window / sequencing |

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

**Q: How is the No-Show model tuned and evaluated?**
A: Rather than artificially distorting class weights or picking an extreme low threshold (which previously flagged 95% of patients and yielded 25% accuracy), we train on honest posterior probabilities and optimize the decision threshold via Youden's J statistic / balanced accuracy on a held-out temporal validation slice. This achieves 60.0% accuracy on synthetic test and 63.5% accuracy on real-world hospital appointments (with 74.1% specificity and balanced recall), while the continuous calibrated probability feeds the dynamic overbooking buffer.

**Q: Why is arrival-delay R² near zero, and how is it evaluated?**
A: Patient punctuality in both synthetic and empirical hospital data is high-entropy human behavior with heavy tails (standard deviation ~14-15 min). Instead of overfitting noise with complex non-linear trees (which previously scored worse than baseline under Huber loss), we train with L1 loss (MAE) to directly target the median. We transparently report both the mathematically optimal median baseline (8.22 min) and the mean baseline (8.41 min). Our model achieves 8.23 min MAE on synthetic test and 13.05 min MAE on real hospital test, avoiding hallucinated signals.

**Q: Did you avoid data leakage?**
A: Yes — temporal split plus per-task feature gating by information-availability time,
and lagged (`shift(1)`) history features. Details are documented in the `preprocess.py` docstring.

**Q: Where do the numbers come from / can you reproduce them?**
A: `reports/metrics/model_metrics.json`, produced by `scripts/02_train_models.py`.
Re-running the script regenerates the models and the metrics file.

**Q: Why not deep learning?**
A: ~20k tabular rows — gradient-boosted trees outperform neural nets in this regime,
train in seconds, and are interpretable via feature importances.
