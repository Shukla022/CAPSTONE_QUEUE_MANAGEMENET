"""Test ML-based slot creation, auto-reopening when doctor finishes all slots,
and clean appointment booking without false conflicts.
"""
import datetime
from pathlib import Path
import sys
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

import scheduling as sc
import service
import store


@pytest.fixture
def clean_test_env():
    """Isolated environment with authentic doctors and duty published."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test_slots.db"
        store.init_db(db)
        orig_db_path = service.db_path
        service.db_path = lambda: Path(db)

        # Seed standard authentic doctors & users
        store.seed_default_users_and_doctors(db)

        # Publish duty for Doctor 1 (Dr. Rajesh Sharma, Cardiology, 09:00 - 13:00)
        today = datetime.date.today().isoformat()
        store.publish_doctor_duty(
            db, 1, today,
            is_available=True, shift_start="09:00", shift_end="13:00",
            published_by="doctor:dr_sharma"
        )

        yield {
            "db": db,
            "today": today,
            "doctor_id": 1,
        }

        service.db_path = orig_db_path


def test_ml_based_slot_creation(clean_test_env):
    """Slot creation must use ML predicted service duration and dynamic buffer."""
    doc = service.get_doctor(clean_test_env["doctor_id"])
    today = clean_test_env["today"]

    # Generate available slots
    slots = service.available_slots(doc, today)
    assert len(slots) > 0

    # Cardiology predicted service duration is ~33.5 min
    first_slot = slots[0]
    assert first_slot.get("slot_start") == "09:00"
    svc_min = first_slot.get("service_minutes")
    assert svc_min is not None
    # ML predicted service time for Cardiology is approx 33.5 min
    assert 30.0 <= svc_min <= 40.0
    assert first_slot.get("buffer_minutes") is not None


def test_auto_reopen_slots_when_doctor_finishes_all_slots(clean_test_env):
    """When a doctor finishes all booked slots and duty time remains,

    it must automatically create new booking slots open.
    """
    db = clean_test_env["db"]
    today = clean_test_env["today"]
    doc = service.get_doctor(clean_test_env["doctor_id"])

    # 1. Book initial slots to fill the schedule
    initial_slots = service.available_slots(doc, today)
    assert len(initial_slots) >= 2

    # Book all initial slots
    for idx, s in enumerate(initial_slots):
        b_id, err = service.book_appointment({
            "patient_name": f"Patient_{idx}",
            "patient_user_id": 100 + idx,
            "department": doc["department"],
            "doctor_id": doc["id"],
            "appointment_date": today,
            "time_slot": "Morning",
            "slot_start": s["slot_start"],
            "slot_end": s["slot_end"],
            "status": "booked",
        })
        assert err is None
        assert b_id is not None

    # Now verify all slots are booked (0 slots available before completion)
    slots_mid = service.available_slots(doc, today)
    assert len(slots_mid) == 0

    # 2. Doctor finishes ALL booked slots early (e.g. at 10:00 AM)
    all_bookings = store.list_bookings(db, doctor_id=doc["id"], date=today)
    for b in all_bookings:
        store.complete_service(db, b["id"], service_minutes=15.0, notes="Completed quickly")

    # Verify no active bookings remain in queue
    active = [b for b in store.list_bookings(db, doctor_id=doc["id"], date=today)
              if b.get("status") in ("booked", "checked_in", "in_service")]
    assert len(active) == 0

    # 3. Automatic reopening: service remains until 13:00!
    # Mocking current time to 10:15 AM (615 minutes)
    orig_now = service._now_min
    try:
        service._now_min = lambda: 615.0  # 10:15 AM

        reopened_slots = service.available_slots(doc, today)
        # Should automatically create new open slots!
        assert len(reopened_slots) > 0

        # The slots should be in the remaining duty time (between 10:15 and 13:00)
        for s in reopened_slots:
            sm = sc.to_min(s["slot_start"])
            em = sc.to_min(s["slot_end"])
            assert sm >= 614.0
            assert em <= 780.0  # 13:00 is 780 min

        # 4. A new patient books one of the newly opened slots!
        first_reopened = reopened_slots[0]
        new_b_id, err = service.book_appointment({
            "patient_name": "New Walk-in Patient",
            "patient_user_id": 999,
            "department": doc["department"],
            "doctor_id": doc["id"],
            "appointment_date": today,
            "time_slot": "Morning",
            "slot_start": first_reopened["slot_start"],
            "slot_end": first_reopened["slot_end"],
            "status": "booked",
        })
        # Booking must succeed without false conflict against done visits!
        assert err is None
        assert new_b_id is not None

        # The booked appointment is active in the database
        booked_appt = store.get_booking(db, new_b_id)
        assert booked_appt["status"] == "booked"
        assert booked_appt["patient_name"] == "New Walk-in Patient"

    finally:
        service._now_min = orig_now


def test_no_seeded_appointments_in_clean_store(clean_test_env):
    """Clean database must have 0 seeded appointments so booking is not full."""
    db = clean_test_env["db"]
    bookings = store.list_bookings(db)
    assert len(bookings) == 0
