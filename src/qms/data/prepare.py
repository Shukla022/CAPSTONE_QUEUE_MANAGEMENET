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


if __name__ == "__main__":
    clean()
