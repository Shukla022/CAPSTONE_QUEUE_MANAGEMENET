"""Regression tests for the 7 issues in the AI Healthcare Queue Management System:

Issue 1: AI Queue Copilot displays raw appointment data (sanitization and booking guidance).
Issue 2: Completed past appointments appear in the Live Queue.
Issue 3: Patients can request emergency priority while booking.
Issue 4: Queue estimates conflict with the doctor's shift (Dr. Rajesh Sharma 00:30-03:00).
Issue 5: Patients cannot clearly switch between multiple appointments.
Issue 6: Waiting time stays stale after a patient completes consultation.
Issue 7: Booking form retains old data after a successful booking.
"""
from __future__ import annotations

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
from qms.genai.explainer import GenAIExplainer


@pytest.fixture
def isolated_qms():
    """Temporary SQLite DB with seeded doctors and clean service context."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test_issues.db"
        store.init_db(db)
        orig_db_path = service.db_path
        service.db_path = lambda: Path(db)

        # Doctor 1: Dr. Ananya Rao (Dermatology, 10:00-14:00)
        d1 = store.add_doctor(db, {
            "name": "Dr. Ananya Rao",
            "department": "Dermatology",
            "shift_start": "10:00",
            "shift_end": "14:00",
            "slot_minutes": 15,
            "buffer_minutes": 5,
            "active": 1,
        })

        # Doctor 2: Dr. Rajesh Sharma (General Medicine, 00:30-03:00)
        d2 = store.add_doctor(db, {
            "name": "Dr. Rajesh Sharma",
            "department": "General Medicine",
            "shift_start": "00:30",
            "shift_end": "03:00",
            "slot_minutes": 15,
            "buffer_minutes": 5,
            "active": 1,
        })

        # Doctor 3: Dr. Priya Patel (Pediatrics, 09:00-13:00)
        d3 = store.add_doctor(db, {
            "name": "Dr. Priya Patel",
            "department": "Pediatrics",
            "shift_start": "09:00",
            "shift_end": "13:00",
            "slot_minutes": 20,
            "buffer_minutes": 5,
            "active": 1,
        })

        # Publish duty for today ± a few days (doctor-driven availability model)
        today = datetime.date.today()
        for did, start, end in (
            (d1, "10:00", "14:00"),
            (d2, "00:30", "03:00"),
            (d3, "09:00", "13:00"),
        ):
            for offset in range(-1, 8):
                store.publish_doctor_duty(
                    db, did, (today + datetime.timedelta(days=offset)).isoformat(),
                    is_available=True, shift_start=start, shift_end=end,
                    published_by="test", update_default_shift=False,
                )

        yield {
            "db": db,
            "d1": d1,
            "d2": d2,
            "d3": d3,
        }

        service.db_path = orig_db_path


# ===========================================================================
# Issue 1: Copilot Sanitization and Booking Guidance
# ===========================================================================
def test_issue_1_copilot_booking_redirection():
    """Copilot directs booking requests to Book Appointment tab and avoids raw data."""
    explainer = GenAIExplainer()
    context = {
        "patient_name": "Risha",
        "appointment": {
            "id": 10,
            "patient_user_id": 99,
            "doctor_id": 18,
            "noshow_prob": 0.42,
            "check_in_time": "10:00:00",
            "doctor_name": "Dr. Ananya Rao",
            "department": "Dermatology",
            "status": "booked",
        },
    }

    ans = explainer.explain_patient_copilot("book appointment for dermatology", context)
    assert "Book Appointment" in ans or "book" in ans.lower()
    # Ensure sensitive internal fields are NOT leaked in output
    assert "patient_user_id" not in ans
    assert "doctor_id" not in ans
    assert "noshow_prob" not in ans
    assert "0.42" not in ans
    assert "{" not in ans  # No raw serialized JSON/dict


def test_issue_1_copilot_queue_question_sanitized_and_readable():
    """Copilot answers queue questions cleanly without leaking raw dicts."""
    explainer = GenAIExplainer()
    context = {
        "patient_name": "Risha",
        "appointment": {
            "patient_user_id": 12,
            "doctor_id": 4,
            "doctor_name": "Dr. Ananya Rao",
            "department": "Dermatology",
            "status": "checked_in",
            "queue_position": "#1",
            "patients_ahead": 0,
            "estimated_wait": "0 minutes",
            "estimated_turn": "10:15 AM",
            "noshow_prob": 0.123,
            "service_minutes": 15.0,
        },
    }

    ans = explainer.explain_patient_copilot("What is my queue status and wait time?", context)
    assert "{" not in ans
    assert "patient_user_id" not in ans
    assert "doctor_id" not in ans
    assert "noshow_prob" not in ans
    assert "Dr. Ananya Rao" in ans or "checked in" in ans.lower()


# ===========================================================================
# Issue 2: Completed Past Appointments Excluded from Live Queue
# ===========================================================================
def test_issue_2_completed_and_past_appointments_excluded(isolated_qms):
    """Completed appointments and past dates must not appear in the active live queue ticket."""
    db = isolated_qms["db"]
    d1 = isolated_qms["d1"]
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    # Past completed booking
    store.add_booking(db, {
        "patient_name": "Alice",
        "patient_user_id": 101,
        "doctor_id": d1,
        "department": "Dermatology",
        "appointment_date": yesterday,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "done",
    })

    # Today's completed booking
    store.add_booking(db, {
        "patient_name": "Alice",
        "patient_user_id": 101,
        "doctor_id": d1,
        "department": "Dermatology",
        "appointment_date": today,
        "slot_start": "10:30",
        "slot_end": "10:45",
        "status": "done",
    })

    # Without explicit booking_id, get_patient_live_ticket must return None for active queue
    ticket = service.get_patient_live_ticket(patient_user_id=101)
    assert ticket is None

    # Now Alice books an active future appointment for today
    active_b_id = store.add_booking(db, {
        "patient_name": "Alice",
        "patient_user_id": 101,
        "doctor_id": d1,
        "department": "Dermatology",
        "appointment_date": today,
        "slot_start": "11:00",
        "slot_end": "11:15",
        "status": "booked",
    })

    # Now get_patient_live_ticket returns the active appointment
    ticket2 = service.get_patient_live_ticket(patient_user_id=101)
    assert ticket2 is not None
    assert ticket2["id"] == active_b_id
    assert ticket2["status"] == "booked"


# ===========================================================================
# Issue 3: Emergency Priority Request Authorization
# ===========================================================================
def test_issue_3_patient_emergency_booking_rejected(isolated_qms):
    """Patients cannot request emergency priority via regular booking."""
    d1 = isolated_qms["d1"]
    today = datetime.date.today().isoformat()

    # Attempt booking with priority_level = 1
    booking_payload = {
        "patient_name": "Bob",
        "patient_user_id": 102,
        "doctor_id": d1,
        "department": "Dermatology",
        "appointment_date": today,
        "slot_start": "11:00",
        "slot_end": "11:15",
        "priority_level": 1,
        "emergency_reason": "High fever",
        "status": "booked",
    }

    bid, err = service.book_appointment(booking_payload)
    assert bid is None
    assert "Emergency priority cannot be requested" in err

    # Regular booking succeeds
    booking_payload["priority_level"] = 0
    booking_payload["emergency_reason"] = ""
    bid2, err2 = service.book_appointment(booking_payload)
    assert bid2 is not None
    assert err2 is None


def test_issue_3_admin_emergency_walkin_authorized(isolated_qms):
    """Only admins can register emergency walk-ins."""
    d1 = isolated_qms["d1"]
    today = datetime.date.today().isoformat()

    admin_user = {"id": 1, "username": "admin", "role": "admin"}
    patient_user = {"id": 102, "username": "bob", "role": "patient"}

    # Patient fails
    with pytest.raises(PermissionError):
        service.add_emergency_walk_in(
            admin_user=patient_user,
            patient_name="Trauma Patient",
            department="Dermatology",
            doctor_id=d1,
            emergency_reason="Critical trauma",
        )

    # Admin succeeds
    bid, err = service.add_emergency_walk_in(
        admin_user=admin_user,
        patient_name="Trauma Patient",
        department="Dermatology",
        doctor_id=d1,
        emergency_reason="Critical trauma",
    )
    assert err is None
    assert bid is not None

    b = store.get_booking(isolated_qms["db"], bid)
    assert b["priority_level"] == 1
    assert b["emergency_reason"] == "Critical trauma"


# ===========================================================================
# Issue 4: Shift Boundary Queue Estimates (Dr. Rajesh Sharma 00:30-03:00)
# ===========================================================================
def test_issue_4_doctor_shift_boundary_calculations(isolated_qms):
    """Calculations for early-morning shift (00:30-03:00) remain consistent with shift hours."""
    db = isolated_qms["db"]
    d2 = isolated_qms["d2"]  # Dr. Rajesh Sharma (00:30-03:00)
    today = datetime.date.today().isoformat()

    # Appointment at 00:30
    bid = store.add_booking(db, {
        "patient_name": "Late Night Patient",
        "patient_user_id": 103,
        "doctor_id": d2,
        "department": "General Medicine",
        "appointment_date": today,
        "slot_start": "00:30",
        "slot_end": "00:45",
        "status": "booked",
    })

    doc = store.get_doctor(db, d2)

    # Test timeline calculation at simulated time 01:00 (during shift)
    sched = sc.live_schedule(doc, store.list_bookings(db, doctor_id=d2, date=today), now_min=60)
    assert "12:30 AM" in sched["doctor"]["shift_12h"]
    assert "3:00 AM" in sched["doctor"]["shift_12h"]
    assert sched["summary"]["shift_end"] == "03:00"
    # Projected finish at 01:15 is well within shift (shift ends at 03:00)
    assert sched["summary"]["projected_finish"] == "01:15"

    # Test timeline calculation when current time is 10:50 AM (now_min = 650, shift ended at 03:00)
    sched_ended = sc.live_schedule(doc, store.list_bookings(db, doctor_id=d2, date=today), now_min=650)
    assert sched_ended["summary"]["shift_end"] == "03:00"
    # Verify projected finish does not explode to 12:01 PM (721 min)
    assert "Shift Concluded" in sched_ended["summary"]["projected_finish_12h"]
    assert sched_ended["summary"]["projected_finish"] < "04:00"


# ===========================================================================
# Issue 5: Patients Switching Between Multiple Appointments
# ===========================================================================
def test_issue_5_switching_multiple_patient_appointments(isolated_qms, monkeypatch):
    """Patients can view and select specific appointments independently."""
    db = isolated_qms["db"]
    d1 = isolated_qms["d1"]
    d3 = isolated_qms["d3"]
    today = datetime.date.today().isoformat()

    # Mock now_min to 10:05 AM (605 min) so it falls within doctor's shift
    monkeypatch.setattr(service, "_now_min", lambda: 605.0)

    # Risha has 2 bookings today: one with Dr. Ananya Rao, one with Dr. Priya Patel
    bid_derma = store.add_booking(db, {
        "patient_name": "Risha",
        "patient_user_id": 200,
        "doctor_id": d1,
        "department": "Dermatology",
        "appointment_date": today,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })

    bid_pedia = store.add_booking(db, {
        "patient_name": "Risha",
        "patient_user_id": 200,
        "doctor_id": d3,
        "department": "Pediatrics",
        "appointment_date": today,
        "slot_start": "11:00",
        "slot_end": "11:20",
        "status": "booked",
    })

    # Fetch specifically Dermatology booking
    ticket_derma = service.get_patient_live_ticket(patient_user_id=200, booking_id=bid_derma)
    assert ticket_derma is not None
    assert ticket_derma["id"] == bid_derma
    assert ticket_derma["doctor_department"] == "Dermatology"

    # Fetch specifically Pediatrics booking
    ticket_pedia = service.get_patient_live_ticket(patient_user_id=200, booking_id=bid_pedia)
    assert ticket_pedia is not None
    assert ticket_pedia["id"] == bid_pedia
    assert ticket_pedia["doctor_department"] == "Pediatrics"

    # Check in for Dermatology
    risha_user = {"id": 200, "username": "risha", "role": "patient"}
    service.patient_check_in(risha_user, bid_derma)

    # Verify Dermatology is checked in, but Pediatrics remains booked
    b_d = store.get_booking(db, bid_derma)
    b_p = store.get_booking(db, bid_pedia)
    assert b_d["status"] == "checked_in"
    assert b_p["status"] == "booked"


# ===========================================================================
# Issue 6: Waiting Time Recalculation When Patient Completes
# ===========================================================================
def test_issue_6_waiting_time_recalculates_on_completion(isolated_qms, monkeypatch):
    """When a patient completes consultation and doctor is free, #1 waiting patient wait is 0 min."""
    db = isolated_qms["db"]
    d1 = isolated_qms["d1"]
    today = datetime.date.today().isoformat()

    # Simulated clock at 10:00 AM when Eisha starts
    sim_clock = {"now": 600.0}
    monkeypatch.setattr(service, "_now_min", lambda: sim_clock["now"])

    # Patient 1: Eisha (10:00 - 10:15)
    bid_eisha = store.add_booking(db, {
        "patient_name": "Eisha",
        "patient_user_id": 301,
        "doctor_id": d1,
        "department": "Dermatology",
        "appointment_date": today,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })

    # Patient 2: Prisha (10:15 - 10:30)
    bid_prisha = store.add_booking(db, {
        "patient_name": "Prisha",
        "patient_user_id": 302,
        "doctor_id": d1,
        "department": "Dermatology",
        "appointment_date": today,
        "slot_start": "10:15",
        "slot_end": "10:30",
        "status": "booked",
    })

    doc_user = {"id": 10, "username": "dr_ananya", "role": "doctor"}

    # Check in both
    store.check_in(db, bid_eisha)
    store.check_in(db, bid_prisha)

    # Start Eisha at 10:00 AM
    service.doctor_start_service(doc_user, bid_eisha)
    b_eisha = store.get_booking(db, bid_eisha)
    assert b_eisha["status"] == "in_service"

    # Advance clock to 10:05 AM (midway through Eisha's consultation)
    sim_clock["now"] = 605.0
    ticket_prisha_waiting = service.get_patient_live_ticket(patient_user_id=302, booking_id=bid_prisha)
    assert ticket_prisha_waiting["patients_ahead"] == 1
    assert ticket_prisha_waiting["live_eta_minutes"] > 0

    # Advance clock to 10:15 AM: Doctor completes Eisha after 15 min consultation
    sim_clock["now"] = 615.0
    service.doctor_complete_service(doc_user, bid_eisha, service_minutes=15.0, notes="Prescribed ointment")
    b_eisha_done = store.get_booking(db, bid_eisha)
    assert b_eisha_done["status"] == "done"

    # Now Prisha is position #1, 0 patients ahead, doctor is free at 10:15 AM
    ticket_prisha_next = service.get_patient_live_ticket(patient_user_id=302, booking_id=bid_prisha)
    assert ticket_prisha_next["patients_ahead"] == 0
    assert ticket_prisha_next["queue_position"] == "#1"
    # Wait time must be 0 minutes, not stale 11-12 min!
    assert ticket_prisha_next["live_eta_minutes"] == 0.0
    assert ticket_prisha_next["live_eta_display"] == "0 minutes"


# ===========================================================================
# Issue 7: Booking Form Reset Behavior and Input Preservation
# ===========================================================================
def test_issue_7_booking_validation_and_preservation(isolated_qms):
    """Booking validation failure preserves inputs; successful booking creates appointment safely."""
    d1 = isolated_qms["d1"]
    today = datetime.date.today().isoformat()

    # Invalid doctor ID
    invalid_booking = {
        "patient_name": "Test User",
        "patient_user_id": 401,
        "doctor_id": 99999,
        "department": "Dermatology",
        "appointment_date": today,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "priority_level": 0,
        "emergency_reason": "",
        "status": "booked",
    }

    bid, err = service.book_appointment(invalid_booking)
    # Validation failed -> Error returned so UI preserves current inputs
    assert bid is None
    assert err is not None

    # Valid doctor ID
    valid_booking = dict(invalid_booking)
    valid_booking["doctor_id"] = d1
    bid2, err2 = service.book_appointment(valid_booking)
    # Succeeded -> booking created without altering or deleting database records
    assert bid2 is not None
    assert err2 is None

    saved_b = store.get_booking(isolated_qms["db"], bid2)
    assert saved_b["patient_name"] == "Test User"
    assert saved_b["status"] == "booked"
