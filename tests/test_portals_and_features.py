"""Comprehensive tests for Patient Live Queue and Doctor Live Queue predictions.

Covers Test Cases A through F from user specifications:
- TEST A: Future booked appointment
- TEST B: Patient checks in
- TEST C: Multiple patients (Ravi, Prisha, Ananya)
- TEST D: Start consultation
- TEST E: Complete consultation
- TEST F: Future appointment separated from live queue
- TEST G: Doctor Portal metrics, projected finish, and negative slack (overrun)
- TEST H: Doctor availability on selected dates
"""
from __future__ import annotations

import datetime
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

import scheduling as sc
import service
import store


def test_case_a_future_booked_appointment(tmp_path=None, monkeypatch=None):
    """TEST A: Future booked appointment must not be in active live queue,
    must not show #1 or ETA=0, must show 'Available after check-in' and turn=slot time.
    """
    import shutil
    if tmp_path is not None and monkeypatch is not None:
        isolated_db = tmp_path / "test_a_qms.db"
        shutil.copy2(service.db_path(), isolated_db)
        monkeypatch.setattr(service, "db_path", lambda: isolated_db)
        db = isolated_db
    else:
        db = service.db_path()

    # Ensure controlled test state: booking 19 is a future booked appointment for patient 25
    store.update_booking(db, 19, status="booked", check_in_time=None)
    store.update_doctor(db, 20, shift_start="10:00", shift_end="14:00")

    # Test on controlled database record (booking id 19, patient_user_id 25)
    ticket = service.get_patient_live_ticket(25, booking_id=19)
    assert ticket is not None, "Patient 25 ticket should exist"
    assert ticket["id"] == 19
    assert ticket["status"] == "booked"
    assert ticket["appointment_date"] == "2026-10-04"
    assert ticket["slot_start_12h"] == "10:00 AM"

    # Strict user requirements:
    # 1. Do NOT treat patient as part of live waiting queue
    assert ticket["is_live_queue"] is False, "Future booked appointment must not be in live queue"
    # 2. Do NOT show Queue Position #1
    assert ticket["queue_position"] == "Available after check-in"
    assert ticket["queue_position_display"] == "Available after check-in"
    assert ticket["queue_pos_num"] is None
    # 3. Do NOT show Live ETA = 0 min
    assert ticket["live_eta_minutes"] is None
    assert ticket["live_eta_display"] == "Available after check-in"
    # 4. Estimated turn must be the scheduled slot time
    assert ticket["estimated_turn"] == "10:00 AM"
    assert ticket["estimated_turn_display"] == "10:00 AM"
    # 5. Doctor shift displayed
    assert ticket["doctor_shift"] == "10:00 AM - 2:00 PM"
    assert ticket["appointment_date_human"] == "October 4, 2026"

    print("[PASS] TEST A: Future booked appointment verified.")


def test_queue_lifecycle_tests_b_c_d_e():
    """TESTS B, C, D, E:
    Simulates a doctor shift with patients Ravi, Prisha, Ananya.
    Tests check-in, multi-patient ordering, consultation start, and completion.
    """
    doc = {
        "id": 101,
        "name": "Dr. Ananya Rao",
        "department": "Dermatology",
        "shift_start": "10:00",
        "shift_end": "14:00",
        "slot_minutes": 12,
        "buffer_minutes": 0,
    }

    # Custom patient durations via remaining_service_fn
    durations = {"Ravi": 15.0, "Prisha": 20.0, "Ananya": 15.0}

    def dur_fn(at_min, b=None):
        if not b:
            return 12.0
        name = b.get("patient_name")
        return durations.get(name, 12.0)

    # =========================================================================
    # TEST B: Single patient (Ravi) checks in at 10:00
    # =========================================================================
    bookings_b = [
        {
            "id": 1,
            "patient_name": "Ravi",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:00:00",
            "slot_start": "10:00",
            "slot_end": "10:12",
            "noshow_prob": 0.05,
        }
    ]

    sched_b = sc.live_schedule(
        doc,
        bookings_b,
        now_min=sc.to_min("10:00"),
        is_today=True,
        remaining_service_fn=dur_fn,
    )

    items_b = sched_b["timeline"]
    assert len(items_b) == 1
    ravi_b = items_b[0]
    assert ravi_b["status"] == "checked_in"
    assert ravi_b["queue_position"] == "#1"
    assert ravi_b["patients_ahead"] == 0
    assert ravi_b["est_wait_min"] == 0.0
    assert ravi_b["est_turn"] == "10:00 AM"

    summary_b = sched_b["summary"]
    assert summary_b["checked_in_count"] == 1
    assert summary_b["booked_count"] == 0
    assert summary_b["in_service_count"] == 0
    print("[PASS] TEST B: Patient check-in verified.")

    # =========================================================================
    # TEST C: Multiple patients (Ravi, Prisha, Ananya) all checked in
    # =========================================================================
    bookings_c = [
        {
            "id": 1,
            "patient_name": "Ravi",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:00:00",
            "slot_start": "10:00",
            "slot_end": "10:12",
            "noshow_prob": 0.05,
        },
        {
            "id": 2,
            "patient_name": "Prisha",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:02:00",
            "slot_start": "10:12",
            "slot_end": "10:24",
            "noshow_prob": 0.05,
        },
        {
            "id": 3,
            "patient_name": "Ananya",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:04:00",
            "slot_start": "10:24",
            "slot_end": "10:36",
            "noshow_prob": 0.05,
        },
    ]

    sched_c = sc.live_schedule(
        doc,
        bookings_c,
        now_min=sc.to_min("10:00"),
        is_today=True,
        remaining_service_fn=dur_fn,
    )

    items_c = {it["patient_name"]: it for it in sched_c["timeline"]}

    # Ravi is #1
    assert items_c["Ravi"]["queue_position"] == "#1"
    assert items_c["Ravi"]["patients_ahead"] == 0
    assert items_c["Ravi"]["est_wait_min"] == 0.0
    assert items_c["Ravi"]["est_turn"] == "10:00 AM"

    # Prisha is #2, waiting for Ravi (15 min)
    assert items_c["Prisha"]["queue_position"] == "#2"
    assert items_c["Prisha"]["patients_ahead"] == 1
    assert items_c["Prisha"]["est_wait_min"] == 15.0
    assert items_c["Prisha"]["est_turn"] == "10:15 AM"

    # Ananya is #3, waiting for Ravi (15 min) + Prisha (20 min) = 35 min
    assert items_c["Ananya"]["queue_position"] == "#3"
    assert items_c["Ananya"]["patients_ahead"] == 2
    assert items_c["Ananya"]["est_wait_min"] == 35.0
    assert items_c["Ananya"]["est_turn"] == "10:35 AM"

    summary_c = sched_c["summary"]
    assert summary_c["checked_in_count"] == 3
    assert summary_c["booked_count"] == 0
    print("[PASS] TEST C: Multiple patients (Ravi #1, Prisha #2, Ananya #3) verified.")

    # =========================================================================
    # TEST D: Start Ravi's consultation at 10:05
    # =========================================================================
    # Ravi is now in_service with service_start at 10:05
    bookings_d = [
        {
            "id": 1,
            "patient_name": "Ravi",
            "priority_level": 0,
            "status": "in_service",
            "service_start": "2026-10-03T10:05:00",
            "check_in_time": "2026-10-03T10:00:00",
            "slot_start": "10:00",
            "slot_end": "10:12",
            "noshow_prob": 0.05,
        },
        {
            "id": 2,
            "patient_name": "Prisha",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:02:00",
            "slot_start": "10:12",
            "slot_end": "10:24",
            "noshow_prob": 0.05,
        },
        {
            "id": 3,
            "patient_name": "Ananya",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:04:00",
            "slot_start": "10:24",
            "slot_end": "10:36",
            "noshow_prob": 0.05,
        },
    ]

    # Current time is 10:10 (5 minutes into Ravi's 15 min consultation, so 10 min remain)
    sched_d = sc.live_schedule(
        doc,
        bookings_d,
        now_min=sc.to_min("10:10"),
        is_today=True,
        remaining_service_fn=dur_fn,
    )

    items_d = {it["patient_name"]: it for it in sched_d["timeline"]}

    # Ravi is In Consultation
    assert items_d["Ravi"]["queue_position"] == "Current Patient"
    assert items_d["Ravi"]["est_turn"] == "Now"
    assert items_d["Ravi"]["est_wait_min"] == 0.0

    # Prisha is #2 (Ravi in consultation ahead of her)
    # Expected wait = remaining time of Ravi = 10 min
    assert items_d["Prisha"]["queue_position"] == "#2"
    assert items_d["Prisha"]["patients_ahead"] == 1  # 1 patient currently being served ahead
    assert items_d["Prisha"]["est_wait_min"] == 10.0
    assert items_d["Prisha"]["est_turn"] == "10:20 AM"

    # Ananya is #3 (Ravi + Prisha ahead of her)
    # Expected wait = remaining of Ravi (10 min) + Prisha duration (20 min) = 30 min
    assert items_d["Ananya"]["queue_position"] == "#3"
    assert items_d["Ananya"]["patients_ahead"] == 2
    assert items_d["Ananya"]["est_wait_min"] == 30.0
    assert items_d["Ananya"]["est_turn"] == "10:40 AM"

    summary_d = sched_d["summary"]
    assert summary_d["in_service_count"] == 1
    assert summary_d["checked_in_count"] == 2
    print("[PASS] TEST D: Start consultation verified (Ravi current, Prisha #2 with 10m ETA, Ananya #3 with 30m ETA).")

    # =========================================================================
    # TEST E: Complete Ravi's consultation at 10:18
    # =========================================================================
    # Ravi completed taking 13 minutes (finished early)
    bookings_e = [
        {
            "id": 1,
            "patient_name": "Ravi",
            "priority_level": 0,
            "status": "done",
            "service_start": "2026-10-03T10:05:00",
            "service_minutes": 13.0,
            "check_in_time": "2026-10-03T10:00:00",
            "slot_start": "10:00",
            "slot_end": "10:12",
            "noshow_prob": 0.05,
        },
        {
            "id": 2,
            "patient_name": "Prisha",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:02:00",
            "slot_start": "10:12",
            "slot_end": "10:24",
            "noshow_prob": 0.05,
        },
        {
            "id": 3,
            "patient_name": "Ananya",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-03T10:04:00",
            "slot_start": "10:24",
            "slot_end": "10:36",
            "noshow_prob": 0.05,
        },
    ]

    # Current time is 10:18 (immediately when Ravi finishes)
    sched_e = sc.live_schedule(
        doc,
        bookings_e,
        now_min=sc.to_min("10:18"),
        is_today=True,
        remaining_service_fn=dur_fn,
    )

    items_e = {it["patient_name"]: it for it in sched_e["timeline"]}

    # Ravi is Completed
    assert items_e["Ravi"]["queue_position"] == "Completed"
    assert items_e["Ravi"]["est_turn"] == "Completed"

    # Prisha moves to #1 with 0 patients ahead, turn is now (10:18 AM)
    assert items_e["Prisha"]["queue_position"] == "#1"
    assert items_e["Prisha"]["patients_ahead"] == 0
    assert items_e["Prisha"]["est_wait_min"] == 0.0
    assert items_e["Prisha"]["est_turn"] == "10:18 AM"

    # Ananya moves to #2 with 1 patient ahead (Prisha: 20 min)
    assert items_e["Ananya"]["queue_position"] == "#2"
    assert items_e["Ananya"]["patients_ahead"] == 1
    assert items_e["Ananya"]["est_wait_min"] == 20.0
    assert items_e["Ananya"]["est_turn"] == "10:38 AM"

    summary_e = sched_e["summary"]
    assert summary_e["done_count"] == 1
    assert summary_e["checked_in_count"] == 2
    assert summary_e["in_service_count"] == 0
    print("[PASS] TEST E: Complete consultation verified (Ravi completed, Prisha #1, Ananya #2).")


def test_case_f_future_appointment_separation():
    """TEST F: Future appointments must appear as upcoming appointments
    and must NOT be part of today's live queue.
    """
    doc = {
        "id": 102,
        "name": "Dr. Ananya Rao",
        "department": "Dermatology",
        "shift_start": "10:00",
        "shift_end": "14:00",
        "slot_minutes": 15,
        "buffer_minutes": 0,
    }

    # Mix of today's active appointments and future appointments
    bookings = [
        # Today's checked-in patient
        {
            "id": 10,
            "patient_name": "Ravi",
            "appointment_date": "2026-10-03",
            "slot_start": "10:00",
            "status": "checked_in",
        },
        # Future appointment for tomorrow
        {
            "id": 11,
            "patient_name": "Ananya",
            "appointment_date": "2026-10-04",
            "slot_start": "10:30",
            "status": "booked",
        },
        # Future appointment for day after tomorrow
        {
            "id": 12,
            "patient_name": "Siddharth",
            "appointment_date": "2026-10-05",
            "slot_start": "11:00",
            "status": "booked",
        },
    ]

    # Filter as done in Doctor Portal:
    today_iso = "2026-10-03"
    today_bookings = [b for b in bookings if (b.get("appointment_date") or today_iso) == today_iso]
    upcoming_bookings = [b for b in bookings if (b.get("appointment_date") or today_iso) > today_iso]

    assert len(today_bookings) == 1
    assert today_bookings[0]["patient_name"] == "Ravi"

    assert len(upcoming_bookings) == 2
    assert [b["patient_name"] for b in upcoming_bookings] == ["Ananya", "Siddharth"]

    # When running live schedule for today:
    sched = sc.live_schedule(doc, today_bookings, now_min=sc.to_min("10:00"), is_today=True)
    assert sched["summary"]["checked_in_count"] == 1
    assert sched["summary"]["booked_count"] == 0

    print("[PASS] TEST F: Future appointment separation verified.")


def test_projected_finish_and_negative_slack():
    """TEST G: Doctor Portal metrics, projected finish, and overrun (negative slack)."""
    doc = {
        "id": 103,
        "name": "Dr. Ananya Rao",
        "shift_start": "10:00",
        "shift_end": "14:00",  # 240 minutes (finishes at 14:00 / 840 min)
        "slot_minutes": 20,
        "buffer_minutes": 0,
    }

    # Severe workload that exceeds the shift end (e.g. 5 patients * 60 min = 300 min from 10:00 -> 15:00)
    heavy_bookings = [
        {"id": i, "patient_name": f"P{i}", "status": "checked_in", "slot_start": "10:00"}
        for i in range(1, 6)
    ]

    sched = sc.live_schedule(
        doc,
        heavy_bookings,
        now_min=sc.to_min("10:00"),
        is_today=True,
        remaining_service_fn=lambda _t, _b: 60.0,
    )

    summary = sched["summary"]
    # 5 * 60 min = 300 min -> 10:00 + 5h = 15:00
    assert summary["projected_finish"] == "15:00"
    assert summary["projected_finish_12h"] == "3:00 PM"
    # Shift end is 14:00 -> Overrun is 60 minutes -> slack_min = -60.0
    assert summary["slack_min"] == -60.0
    assert summary["finishing_early"] is False

    print("[PASS] TEST G: Projected finish (3:00 PM) and negative slack (-60.0 min) verified.")


def test_doctor_availability_enforcement():
    """TEST H: Duty is doctor-published; inactive doctors stay blocked; closed duty blocks slots."""
    db = service.db_path()
    docs = service.list_all_hospital_doctors()
    assert len(docs) > 0
    test_doc = docs[0]
    doc_id = test_doc["id"]
    doc_user = {"role": "doctor", "username": "test_doc", "doctor_id": doc_id}

    # Unpublished date must not be bookable (opt-in publish model)
    unpublished = "2099-01-05"
    status_none, reason_none = service.is_doctor_available_on_date(doc_id, unpublished)
    assert status_none is False
    assert "publish" in reason_none.lower() or "duty" in reason_none.lower()

    # Test inactive doctor
    try:
        store.set_doctor_status(db, doc_id, False)
        status, reason = service.is_doctor_available_on_date(doc_id, "2026-10-04")
        assert status is False
        assert "inactive" in reason.lower()

        sched = service.live_schedule(doc_id, date_iso="2026-10-04")
        assert sched["summary"]["available"] is False
    finally:
        store.set_doctor_status(db, doc_id, True)

    # Doctor publishes duty → slots can open; closing duty removes them
    try:
        service.publish_doctor_duty(
            doc_user, doc_id, "2026-10-15",
            is_available=True, shift_start="09:00", shift_end="13:00",
        )
        status_on, _ = service.is_doctor_available_on_date(doc_id, "2026-10-15")
        assert status_on is True

        service.publish_doctor_duty(
            doc_user, doc_id, "2026-10-15",
            is_available=False, shift_start="09:00", shift_end="13:00",
        )
        status_unavail, reason_unavail = service.is_doctor_available_on_date(doc_id, "2026-10-15")
        assert status_unavail is False
        assert "duty" in reason_unavail.lower() or "publish" in reason_unavail.lower()

        slots = service.available_slots(test_doc, "2026-10-15")
        assert len(slots) == 0, "Slots should be empty when doctor closed duty for the date"
    finally:
        service.publish_doctor_duty(
            doc_user, doc_id, "2026-10-15",
            is_available=True, shift_start="09:00", shift_end="13:00",
        )

    # Publish a known date and confirm active
    service.publish_doctor_duty(
        doc_user, doc_id, "2026-10-04",
        is_available=True, shift_start="09:00", shift_end="13:00",
    )
    status_active, _ = service.is_doctor_available_on_date(doc_id, "2026-10-04")
    assert status_active is True
    print("[PASS] TEST H: Doctor-published duty and inactive enforcement verified.")


if __name__ == "__main__":
    test_case_a_future_booked_appointment()
    test_queue_lifecycle_tests_b_c_d_e()
    test_case_f_future_appointment_separation()
    test_projected_finish_and_negative_slack()
    test_doctor_availability_enforcement()
    print("\nALL VERIFICATION TESTS (A through H) PASSED SUCCESSFULLY!")
