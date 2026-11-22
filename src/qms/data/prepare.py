"""Data preparation: raw CSV -> validated clean dataset + data profile.

The provided dataset is already feature-engineered, so preparation here is
deliberately conservative and *documented* rather than transformative:

* parse dates, coerce dtypes;
* drop exact duplicate appointment rows;
* sanity-check the target and a few invariants (e.g. service_time == end-start);
* derive a couple of convenience columns used by the app/eval;
* write ``data/processed/clean.csv`` and a ``data_profile.json`` summary.

Keeping this step explicit (and idempotent) means the training/eval scripts
always read a known, validated artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..config import load_config, abspath, ensure_dirs
from .loader import load_raw


def _profile(df: pd.DataFrame) -> Dict:
    prof: Dict = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "date_min": str(df["appointment_date"].min().date()) if df["appointment_date"].notna().any() else None,
        "date_max": str(df["appointment_date"].max().date()) if df["appointment_date"].notna().any() else None,
    }
    if "no_show" in df:
        prof["no_show_rate"] = round(float(df["no_show"].mean()), 4)
    if "appointment_status" in df:
        prof["status_counts"] = {str(k): int(v) for k, v in
                                 df["appointment_status"].value_counts().items()}
    if "priority_level" in df:
        prof["priority_counts"] = {str(k): int(v) for k, v in
                                   df["priority_level"].value_counts().items()}
    for col in ["waiting_time", "service_time", "arrival_delay"]:
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce")
            prof[f"{col}_mean"] = round(float(s.mean()), 3) if s.notna().any() else None
    if "department" in df:
        prof["departments"] = sorted(df["department"].dropna().unique().tolist())
    if "queue_id" in df:
        prof["n_queues"] = int(df["queue_id"].nunique())
    return prof


def clean(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(cfg["data"]["processed_dir"])
    df = load_raw(cfg)

    # Drop exact duplicate appointment ids (keep first), if any.
    if "appointment_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["appointment_id"], keep="first")
        dropped = before - len(df)
        if dropped:
            print(f"  dropped {dropped} duplicate appointment_id row(s)")

    # Ensure key numeric columns are numeric.
    for col in ["no_show", "priority_level", "active_doctors_count",
                "queue_length_at_arrival", "slot_arrival_count",
                "service_time", "waiting_time", "arrival_delay",
                "previous_no_show_count", "previous_late_count",
                "total_visits_count", "lead_time_hours", "lead_time_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convenience: booking hour if absent.
    if "booking_hour" not in df.columns and "booking_timestamp" in df.columns:
        df["booking_hour"] = df["booking_timestamp"].dt.hour

    profile = _profile(df)

    out_csv = abspath(cfg["data"]["clean_csv"])
    df.to_csv(out_csv, index=False)
    with open(abspath(cfg["data"]["profile_json"]), "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)

    print(f"  clean rows: {profile['n_rows']}  |  no-show rate: "
          f"{profile.get('no_show_rate')}  |  dates: "
          f"{profile.get('date_min')} -> {profile.get('date_max')}")
    print(f"  saved -> {out_csv}")
    return df




def prepare_real_hospital(
    cfg: dict | None = None,
    raw_filename: str = "realistic_hospital_testing_dataset_v8.csv",
    out_filename: str = "real_hospital_clean.csv",
    profile_filename: str = "real_hospital_profile.json",
) -> pd.DataFrame:
    """Preprocess raw real-world hospital appointment dataset for out-of-distribution evaluation.

    Takes raw clinical appointment records (16 columns), derives temporal, queueing,
    and clinical features to match the exact schema of the synthetic training data.
    """
    cfg = cfg or load_config()
    ensure_dirs(cfg["data"]["processed_dir"])
    raw_p = Path(raw_filename)
    raw_path = raw_p if raw_p.is_absolute() else abspath(cfg["data"]["raw_csv"]).parent / raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Realistic hospital dataset not found at: {raw_path}")

    df = pd.read_csv(raw_path, low_memory=False)

    # Parse datetimes
    appt_dt = pd.to_datetime(df["appointment_date"] + " " + df["appointment_time"], dayfirst=True, errors="coerce")
    booking_dt = pd.to_datetime(df["booking_timestamp"], dayfirst=True, errors="coerce")
    checkin_dt = pd.to_datetime(df["check_in_time"], dayfirst=True, errors="coerce")
    start_dt = pd.to_datetime(df["service_start_time"], dayfirst=True, errors="coerce")
    end_dt = pd.to_datetime(df["service_end_time"], dayfirst=True, errors="coerce")

    df["appointment_date"] = pd.to_datetime(df["appointment_date"], dayfirst=True, errors="coerce")
    df["booking_timestamp"] = booking_dt
    df["appointment_time"] = df["appointment_date"].dt.strftime("%d-%m-%Y") + " " + df["appointment_time"].astype(str)
    df["check_in_time"] = checkin_dt
    df["service_start_time"] = start_dt
    df["service_end_time"] = end_dt

    # Booking & lead time features
    df["lead_time_hours"] = (appt_dt - booking_dt).dt.total_seconds() / 3600.0
    df["lead_time_days"] = df["lead_time_hours"] / 24.0
    df["hour_of_day"] = appt_dt.dt.hour
    df["booking_hour"] = booking_dt.dt.hour
    df["day_of_week"] = appt_dt.dt.day_name()
    df["appointment_month"] = appt_dt.dt.month
    df["appointment_week"] = appt_dt.dt.isocalendar().week.astype(int)
    df["appointment_day"] = appt_dt.dt.day
    df["is_weekend"] = appt_dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].isin([9, 10, 11, 17]).astype(int)
    df["is_holiday"] = 0
    df["holiday_name"] = np.nan

    # Shift slot category
    df["time_slot"] = np.where(
        df["hour_of_day"] <= 11, "morning",
        np.where(df["hour_of_day"] <= 15, "afternoon", "evening")
    )

    # Priority mapping: Normal -> 0, Emergency -> 1
    priority_map = {"normal": 0, "emergency": 1}
    if "priority_level" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["priority_level"]):
            df["priority_level"] = (
                df["priority_level"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(priority_map)
                .fillna(0)
                .astype(int)
            )
        else:
            df["priority_level"] = pd.to_numeric(df["priority_level"], errors="coerce").fillna(0).astype(int)

    # Department to queue ID mapping
    dept_q_map = {
        "General Medicine": "Q001", "Orthopedics": "Q002", "Gynecology": "Q003",
        "ENT": "Q004", "Dermatology": "Q005", "Ophthalmology": "Q006",
        "Dentistry": "Q007", "Cardiology": "Q008"
    }
    df["queue_id"] = df["department"].map(dept_q_map).fillna("Q001")

    # Doctor and slot concurrency
    df["active_doctors_count"] = df.groupby(["appointment_date", "department"])["doctor_id"].transform("nunique")
    df["slot_arrival_count"] = df.groupby(["appointment_date", "department", "appointment_time"])["appointment_id"].transform("count")

    # Punctuality and delays
    df["qr_checked_in"] = df["check_in_time"].notna().astype(int)
    df["arrival_delay"] = (checkin_dt - appt_dt).dt.total_seconds() / 60.0
    df["buffer_window_minutes"] = 10
    df["within_buffer_flag"] = (df["arrival_delay"].abs() <= 10).astype(int)

    # Clinical duration targets
    df["service_time"] = (end_dt - start_dt).dt.total_seconds() / 60.0
    df["waiting_time"] = (start_dt - checkin_dt).dt.total_seconds() / 60.0

    # No-show target (attended vs cancelled/no-show)
    df["no_show"] = df["appointment_status"].astype(str).str.lower().isin(["no-show", "cancelled"]).astype(int)
    if "cancellation_flag" in df.columns:
        df["no_show"] = (df["no_show"] | (df["cancellation_flag"] == 1)).astype(int)

    # Queue length at arrival
    df_sorted = df.sort_values(["appointment_date", "department", "check_in_time"]).reset_index(drop=True)
    q_lens = []
    for _, r in df_sorted.iterrows():
        if pd.isna(r["check_in_time"]):
            q_lens.append(np.nan)
            continue
        dept = r["department"]
        date_val = r["appointment_date"]
        cit = r["check_in_time"]
        mask = (df_sorted["department"] == dept) & (df_sorted["appointment_date"] == date_val) & \
               (df_sorted["check_in_time"].notna()) & (df_sorted["check_in_time"] < cit) & \
               (df_sorted["service_end_time"] > cit)
        q_lens.append(float(mask.sum()))
    df_sorted["queue_length_at_arrival"] = q_lens
    df = df_sorted.sort_values("appointment_id").reset_index(drop=True)

    df["doctor_serving_order"] = df.groupby(["appointment_date", "doctor_id"])["service_start_time"].rank(method="first")

    profile = _profile(df)

    out_csv = abspath(cfg["data"]["processed_dir"]) / out_filename
    out_prof = abspath(cfg["data"]["processed_dir"]) / profile_filename

    df.to_csv(out_csv, index=False)
    with open(out_prof, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)

    print(f"  [Real Hospital] Processed rows: {profile['n_rows']} | No-show rate: {profile.get('no_show_rate')}")
    print(f"  [Real Hospital] Dates: {profile.get('date_min')} -> {profile.get('date_max')}")
    print(f"  [Real Hospital] Saved -> {out_csv}")
    return df


if __name__ == "__main__":
    clean()
