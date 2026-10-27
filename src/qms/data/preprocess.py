"""Leakage-safe feature engineering and temporal splitting.

Design decisions (documented for the paper):

* This is an **appointment-only** system: there are no walk-ins. Every patient
  books ahead, so the priority signal is (a) clinical emergency
  (``priority_level``) and (b) appointment adherence history.

* **No-show** and **arrival-delay** models may use *only* information known at
  BOOKING time. Any column that is realised at/after arrival (check-in time,
  *this visit's* arrival delay, queue length at arrival, waiting/service times,
  serving order) is EXCLUDED for these two tasks to avoid target leakage.
  Patient / department / hour delay history is allowed only as a *lagged*
  expanding mean of **prior** visits.

* The **arrival-delay** target is trained on completed visits only. Rows labelled
  no-show (including "late arrival" no-shows) are a different process and would
  dominate errors if mixed in.

* The **waiting-time** model is a *real-time* estimate produced at CHECK-IN, so
  it may additionally use ``queue_length_at_arrival`` (known once the patient
  is present), but never the patient's own realised service time.

* The **service-time** model is used to *plan* slot widths from historical
  consult duration given department, weekday and time-of-day — never this
  visit's realised wait or service.

* Train/test split is **temporal** (earliest appointments train, latest test) to
  mimic deployment and avoid leaking the future into the past.
"""
from __future__ import annotations

from typing import Tuple, List
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# Known at booking time (leakage-safe for no-show / arrival-delay prediction).
BOOKING_NUMERIC: List[str] = [
    "lead_time_hours", "lead_time_days", "hour_of_day", "booking_hour",
    "is_weekend", "is_peak_hour", "is_holiday", "priority_level",
    "previous_no_show_count", "previous_late_count", "total_visits_count",
    "active_doctors_count", "slot_arrival_count",
    "no_show_rate", "late_rate", "log_lead_time_hours", "is_same_day",
    "hour_sin", "hour_cos",
]
BOOKING_CATEGORICAL: List[str] = ["department", "day_of_week", "time_slot"]

# Extra booking-time signals used only for arrival-delay (punctuality).
ARRIVAL_NUMERIC: List[str] = BOOKING_NUMERIC + [
    "buffer_window_minutes",
    "dept_mean_delay", "hour_mean_delay",
]

# Known at check-in (for the real-time waiting-time estimate).
CHECKIN_NUMERIC: List[str] = [
    "queue_length_at_arrival", "active_doctors_count", "slot_arrival_count",
    "priority_level", "hour_of_day", "is_peak_hour", "is_weekend",
    "queue_per_doctor", "arrivals_per_doctor",
]
CHECKIN_CATEGORICAL: List[str] = ["department", "day_of_week", "time_slot"]

# Typical consult duration planned at booking (no realised wait/service of *this* visit).
SERVICE_NUMERIC: List[str] = [
    "hour_of_day", "is_weekend", "is_peak_hour", "priority_level",
    "active_doctors_count", "slot_arrival_count",
]
SERVICE_CATEGORICAL: List[str] = ["department", "day_of_week", "time_slot"]

# Task registry: (numeric, categorical, target, requires-nonnull-target)
TASKS = {
    "no_show":       (BOOKING_NUMERIC, BOOKING_CATEGORICAL, "no_show",       False),
    "arrival_delay": (ARRIVAL_NUMERIC, BOOKING_CATEGORICAL, "arrival_delay", True),
    "waiting_time":  (CHECKIN_NUMERIC, CHECKIN_CATEGORICAL, "waiting_time",  True),
    "service_time":  (SERVICE_NUMERIC, SERVICE_CATEGORICAL, "service_time",  True),
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe ratios / logs built only from columns already known at that time."""
    out = df.copy()
    vis = pd.to_numeric(out["total_visits_count"], errors="coerce").fillna(0) if "total_visits_count" in out else 0
    vis = pd.Series(vis, index=out.index).clip(lower=0)
    denom = vis.replace(0, 1.0)
    if "previous_no_show_count" in out:
        ns = pd.to_numeric(out["previous_no_show_count"], errors="coerce").fillna(0)
        out["no_show_rate"] = ns / denom
    else:
        out["no_show_rate"] = 0.0
    if "previous_late_count" in out:
        late = pd.to_numeric(out["previous_late_count"], errors="coerce").fillna(0)
        out["late_rate"] = late / denom
    else:
        out["late_rate"] = 0.0
    if "lead_time_hours" in out:
        lt = pd.to_numeric(out["lead_time_hours"], errors="coerce").fillna(0).clip(lower=0)
        out["log_lead_time_hours"] = np.log1p(lt)
    else:
        out["log_lead_time_hours"] = 0.0
    if "lead_time_days" in out:
        ld = pd.to_numeric(out["lead_time_days"], errors="coerce").fillna(0)
        out["is_same_day"] = (ld <= 0).astype(float)
    else:
        out["is_same_day"] = 0.0
    docs = pd.to_numeric(out["active_doctors_count"], errors="coerce").fillna(1) if "active_doctors_count" in out else 1
    docs = pd.Series(docs, index=out.index).clip(lower=1)
    if "queue_length_at_arrival" in out:
        q = pd.to_numeric(out["queue_length_at_arrival"], errors="coerce").fillna(0)
        out["queue_per_doctor"] = q / docs
    else:
        out["queue_per_doctor"] = 0.0
    if "slot_arrival_count" in out:
        arr = pd.to_numeric(out["slot_arrival_count"], errors="coerce").fillna(0)
        out["arrivals_per_doctor"] = arr / docs
    else:
        out["arrivals_per_doctor"] = 0.0
    if "hour_of_day" in out:
        hour = pd.to_numeric(out["hour_of_day"], errors="coerce").fillna(0.0)
        hour = pd.Series(hour, index=out.index)
    else:
        hour = pd.Series(0.0, index=out.index)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    if "buffer_window_minutes" in out:
        out["buffer_window_minutes"] = pd.to_numeric(
            out["buffer_window_minutes"], errors="coerce").fillna(10.0)
    else:
        out["buffer_window_minutes"] = 10.0
    for col in ("previous_mean_delay", "dept_mean_delay", "hour_mean_delay"):
        if col not in out:
            out[col] = 0.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _lagged_expanding_mean(groups: pd.Series, values: pd.Series) -> pd.Series:
    """Expanding mean of *prior* rows within each group (current row excluded)."""
    return values.groupby(groups, sort=False).transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())


def add_past_delay_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Past-only mean arrival delay for this patient, department, and hour.

    Must be called on chronologically ordered (or will sort) completed visits.
    Uses ``shift(1)`` so this visit's own delay never enters the features.
    """
    out = rows.copy()
    if "arrival_delay" not in out.columns:
        out["previous_mean_delay"] = 0.0
        out["dept_mean_delay"] = 0.0
        out["hour_mean_delay"] = 0.0
        return out

    order_cols = [c for c in ("appointment_date", "booking_timestamp", "appointment_id")
                  if c in out.columns]
    ordered = out.sort_values(order_cols) if order_cols else out
    delay = pd.to_numeric(ordered["arrival_delay"], errors="coerce")
    overall = delay.shift(1).expanding(min_periods=1).mean()

    if "patient_id" in ordered.columns:
        ordered["previous_mean_delay"] = _lagged_expanding_mean(
            ordered["patient_id"], delay)
    else:
        ordered["previous_mean_delay"] = np.nan
    if "department" in ordered.columns:
        ordered["dept_mean_delay"] = _lagged_expanding_mean(
            ordered["department"], delay)
    else:
        ordered["dept_mean_delay"] = np.nan
    if "hour_of_day" in ordered.columns:
        hour_key = pd.to_numeric(ordered["hour_of_day"], errors="coerce").fillna(-1)
        ordered["hour_mean_delay"] = _lagged_expanding_mean(hour_key, delay)
    else:
        ordered["hour_mean_delay"] = np.nan

    for col in ("previous_mean_delay", "dept_mean_delay", "hour_mean_delay"):
        ordered[col] = ordered[col].fillna(overall).fillna(0.0)
        out[col] = ordered[col]
    return out


def get_task_frame(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (raw_features, target, rows) for a given task.

    ``rows`` is the filtered dataframe aligned with the features/target so the
    caller can build a temporal split on ``appointment_date``.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(TASKS)}.")
    numeric, categorical, target, need_target = TASKS[task]

    rows = df.copy()
    if need_target:
        rows = rows[rows[target].notna()].copy()
    if task == "arrival_delay" and "no_show" in rows.columns:
        shown = pd.to_numeric(rows["no_show"], errors="coerce").fillna(0).eq(0)
        rows = rows[shown].copy()
    if task == "arrival_delay":
        rows = add_past_delay_features(rows)
    rows = add_derived_features(rows)

    cols = numeric + categorical
    missing = [c for c in cols + [target] if c not in rows.columns]
    if missing:
        raise KeyError(f"Columns missing for task '{task}': {missing}")

    X_raw = rows[cols].copy()
    # Numeric NaNs (rare) -> 0; categorical NaNs -> 'unknown'
    for c in numeric:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0.0)
    for c in categorical:
        X_raw[c] = X_raw[c].astype("object").fillna("unknown")

    y = rows[target].astype(float)
    return X_raw, y, rows


def fit_encoder(X_raw: pd.DataFrame, categorical: List[str]) -> List[str]:
    """Return the ordered one-hot feature columns learned from training rows."""
    encoded = pd.get_dummies(X_raw, columns=categorical, dummy_na=False)
    return list(encoded.columns)


def transform(X_raw: pd.DataFrame, feature_columns: List[str],
              categorical: List[str]) -> pd.DataFrame:
    """One-hot encode and align to the training feature columns."""
    encoded = pd.get_dummies(X_raw, columns=categorical, dummy_na=False)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded.astype(float)


def temporal_split(rows: pd.DataFrame, test_size: float,
                   date_col: str = "appointment_date") -> np.ndarray:
    """Boolean test mask: the latest ``test_size`` fraction of appointments.

    Sorted by date then booking order so ties are stable.
    """
    order_cols = [date_col]
    if "booking_timestamp" in rows.columns:
        order_cols.append("booking_timestamp")
    elif "appointment_id" in rows.columns:
        order_cols.append("appointment_id")
    order = rows.sort_values(order_cols).index
    n = len(order)
    n_test = int(round(n * test_size))
    test_ids = set(order[n - n_test:])
    return rows.index.isin(test_ids)


def categorical_for(task: str) -> List[str]:
    return TASKS[task][1]
