"""Seed future published duty hours for all demo doctors.

Run once to make the app usable out-of-the-box:
    python scripts/seed_demo_availability.py

Publishes the next 14 days (excluding today if past 14:00) for every active
doctor with a standard 09:00–17:00 shift.
"""
from __future__ import annotations
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "config.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import store  # noqa: E402
import service  # noqa: E402

SHIFT_START = "09:00"
SHIFT_END   = "17:00"
DAYS_AHEAD  = 14

def main():
    store.init_db(service.db_path())
    doctors = service.list_all_hospital_doctors()
    if not doctors:
        print("No doctors found in DB. Run the app first to seed accounts.")
        return

    today = date.today()
    dates = [str(today + timedelta(days=d)) for d in range(1, DAYS_AHEAD + 1)]

    dbp = service.db_path()
    published = 0
    for doc in doctors:
        if not doc.get("active"):
            continue
        did = doc["id"]
        for d in dates:
            already = store.get_doctor_duty(dbp, did, d)
            if already and already.get("is_available"):
                continue  # skip already-published
            store.publish_doctor_duty(
                dbp,
                doctor_id=did,
                available_date=d,
                is_available=True,
                shift_start=SHIFT_START,
                shift_end=SHIFT_END,
            )
            published += 1

    print(f"✅ Published {published} duty records across {len(doctors)} doctors for the next {DAYS_AHEAD} days.")
    print(f"   Shift: {SHIFT_START} – {SHIFT_END}")
    print("   Doctors can override or close individual dates from the Doctor Portal → Shift & Timetable tab.")

if __name__ == "__main__":
    main()
