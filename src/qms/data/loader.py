"""Data loading utilities."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..config import load_config, abspath

# Columns that hold day-first date/time strings in the raw CSV.
_DATETIME_COLS = [
    "booking_timestamp",
    "check_in_time",
    "service_start_time",
    "service_end_time",
]


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    """Load the raw appointment dataset with sensible dtype parsing."""
    cfg = cfg or load_config()
    path = abspath(cfg["data"]["raw_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place "
            "'hospital_dataset_preprocessed.csv' in data/raw/ (see data/raw/README.md)."
        )
    df = pd.read_csv(path, low_memory=False)

    # Parse the appointment date (day-first: DD-MM-YYYY).
    df["appointment_date"] = pd.to_datetime(
        df["appointment_date"], dayfirst=True, errors="coerce"
    )
    for col in _DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df


def load_clean(cfg: dict | None = None) -> pd.DataFrame:
    """Load the cleaned dataset produced by scripts/01_prepare_data.py."""
    cfg = cfg or load_config()
    path = abspath(cfg["data"]["clean_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {path}. Run: python scripts/01_prepare_data.py"
        )
    df = pd.read_csv(path, parse_dates=["appointment_date"], low_memory=False)
    return df
