import sys
from pathlib import Path

app_dir = Path(__file__).resolve().parent.parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import pytest
from datetime import date

import service
import store
import scheduling


def test_patient_checkin_doctor_start_and_complete_sync(tmp_path):
    """Test full synchronized flow across Patient, Doctor, and Admin:
    1. Booking created (booked)
    2. Patient check-in (checked_in)
    3. Doctor starts consultation (in_service)
    4. Doctor completes consultation (done)
    5. Sync checks across patient ticket, doctor schedule, and admin metrics.
    """
    db_file = tmp_path / "test_flow.db"
    store.init_db(db_file)
    store.seed_default_users_and_doctors(db_file)

    today = date.today().isoformat()
    doctors = store.list_doctors(db_file, active_only=True)
    assert doctors, "Doctors must exist"
    doc = doctors[0]

    # Create a patient user
    uid, err = store.create_user(db_file, {
        "username": "flow_patient",
        "password": "pass123",
        "role": "patient",
        "full_name": "Flow Patient",
    })
    assert err is None
    patient = store.authenticate_user(db_file, "flow_patient", "pass123")
    assert patient is not None
    doctor_user = store.authenticate_user(db_file, "dr_priya", "doctor123") or {
        "id": 2, "username": "dr_priya", "role": "doctor"
    }
    admin_user = {"id": 1, "username": "admin", "role": "admin"}

    # 1. Book appointment
    booking_id = store.add_booking(db_file, {
        "doctor_id": doc["id"],
        "patient_name": "Flow Patient",
        "appointment_date": today,
        "slot_start": "10:00",
        "slot_end": "10:20",
        "department": doc["department"],
        "patient_user_id": patient["id"],
        "status": "booked",
    })

    booking = store.get_booking(db_file, booking_id)
    assert booking["status"] == "booked"

    # Admin check initially
    counts = store.counts_by_status(db_file, date=today)
    assert counts.get("booked") == 1
    assert counts.get("checked_in", 0) == 0
    assert counts.get("in_service", 0) == 0
    assert counts.get("done", 0) == 0

    # 2. Patient Check-In
    # Override service.db_path temporarily to test_flow.db
    orig_db_path = service.db_path
    service.db_path = lambda: db_file
    try:
        service.patient_check_in(patient, booking_id)
        booking = store.get_booking(db_file, booking_id)
        assert booking["status"] == "checked_in"
        assert booking["check_in_time"] is not None

        # Patient ticket sync
        ticket = service.get_patient_live_ticket(patient_user_id=patient["id"], booking_id=booking_id)
        assert ticket is not None
        assert ticket["status"] == "checked_in"
        assert ticket["queue_position"] in ("#1", 1)

        # Admin sync
        counts = store.counts_by_status(db_file, date=today)
        assert counts.get("checked_in") == 1
        assert counts.get("in_service", 0) == 0

        # Doctor live schedule sync
        sched = service.live_schedule(doc, today)
        tl = sched.get("timeline", [])
        assert any(t["id"] == booking_id and t["status"] == "checked_in" for t in tl)

        # 3. Doctor starts consultation for this particular patient
        service.doctor_start_service(doctor_user, booking_id)
        booking = store.get_booking(db_file, booking_id)
        assert booking["status"] == "in_service"
        assert booking["service_start"] is not None

        # Patient ticket sync
        ticket = service.get_patient_live_ticket(patient_user_id=patient["id"], booking_id=booking_id)
        assert ticket["status"] == "in_service"
        assert ticket["queue_position"] == "Current Patient"
        assert ticket["estimated_turn"] == "Now"

        # Admin sync
        counts = store.counts_by_status(db_file, date=today)
        assert counts.get("in_service") == 1
        assert counts.get("checked_in", 0) == 0

        # 4. Doctor completes consultation
        res = service.doctor_complete_service(
            doctor_user,
            booking_id,
            service_minutes=12.0,
            notes="Patient treated and discharged with follow up prescription.",
        )
        booking = store.get_booking(db_file, booking_id)
        assert booking["status"] == "done"
        assert booking["service_minutes"] == 12.0
        assert "discharged" in booking["notes"]

        # Patient ticket sync
        ticket = service.get_patient_live_ticket(patient_user_id=patient["id"], booking_id=booking_id)
        assert ticket["status"] == "done"

        # Admin sync
        counts = store.counts_by_status(db_file, date=today)
        assert counts.get("done") == 1
        assert counts.get("in_service", 0) == 0
        assert counts.get("checked_in", 0) == 0

        # Doctor schedule sync: booking is done and no longer in active queue
        sched = service.live_schedule(doc, today)
        tl_active = [t for t in sched.get("timeline", []) if t.get("status") in ("checked_in", "in_service")]
        assert len(tl_active) == 0

    finally:
        service.db_path = orig_db_path
