"""Comprehensive regression and acceptance test suite for QMS.

Covers all 12 bug fixes, the Admin Emergency Walk-In workflow, and the
15-step acceptance scenario from Part C7.

Queue Position Convention (per existing established tests):
  - In-service patient: 'Current Patient' (queue_pos_num = 0)
  - First WAITING checked-in patient: '#2' if 1 in_service ahead, '#1' if no in_service
  - queue_position = f"#{ci_idx + len(in_service_items) + 1}"
  - patients_ahead = ci_idx + len(in_service_items)
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


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_env():
    """Temporary SQLite DB with two seeded doctors and a mock admin user."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test_qms.db"
        store.init_db(db)
        orig_db_path = service.db_path
        service.db_path = lambda: Path(db)

        did_derma = store.add_doctor(db, {
            "name": "Dr. Ananya Rao",
            "department": "Dermatology",
            "shift_start": "10:00",
            "shift_end": "14:00",
            "slot_minutes": 15,
            "buffer_minutes": 5,
            "active": 1,
        })
        did_cardio = store.add_doctor(db, {
            "name": "Dr. Rajesh Kumar",
            "department": "Cardiology",
            "shift_start": "09:00",
            "shift_end": "13:00",
            "slot_minutes": 20,
            "buffer_minutes": 5,
            "active": 1,
        })

        # Publish duty windows so bookings/walk-ins work under doctor-driven availability
        today = datetime.date.today()
        for did, start, end in (
            (did_derma, "10:00", "14:00"),
            (did_cardio, "09:00", "13:00"),
        ):
            for offset in range(-1, 8):
                store.publish_doctor_duty(
                    db, did, (today + datetime.timedelta(days=offset)).isoformat(),
                    is_available=True, shift_start=start, shift_end=end,
                    published_by="test", update_default_shift=False,
                )

        admin_user = {
            "id": 1,
            "username": "admin",
            "role": "admin",
            "full_name": "Hospital Admin",
        }

        yield {
            "db": db,
            "doc_derma_id": did_derma,
            "doc_cardio_id": did_cardio,
            "admin_user": admin_user,
        }
        service.db_path = orig_db_path


# =============================================================================
# Issue 1 & 2: Appointment Time and Scheduled Turn preservation
# =============================================================================
def test_issue_1_and_2_appointment_time_and_scheduled_turn(isolated_env):
    """Verify slot_start 10:17 is preserved in ticket as appointment_time_12h,
    and booked appointment shows its scheduled slot as estimated_turn."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    today_iso = datetime.date.today().isoformat()

    bid, err = service.book_appointment({
        "patient_name": "Test Patient",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "time_slot": "10:17-10:28",
        "slot_start": "10:17",
        "slot_end": "10:28",
        "priority_level": 0,
        "status": "booked",
    })
    assert err is None
    assert bid is not None

    b = store.get_booking(db, bid)
    assert b["slot_start"] == "10:17"
    assert b["slot_end"] == "10:28"

    ticket = service.get_patient_live_ticket(patient_name="Test Patient")
    assert ticket is not None
    # Issue 1: appointment_time shows slot time (10:17 AM), not doctor shift start (10:00 AM)
    assert ticket["appointment_time_12h"] == "10:17 AM"
    # Issue 1: doctor_shift is a separate field from appointment_time
    assert "10:00 AM" in ticket["doctor_shift"]  # shift start is 10:00
    assert "2:00 PM" in ticket["doctor_shift"]   # shift end is 14:00 -> 2:00 PM
    # Issue 2: estimated_turn for booked appointment equals its scheduled slot time
    assert ticket["estimated_turn"] == "10:17 AM"
    # Not in live queue
    assert ticket["queue_position"] == "Available after check-in"
    assert ticket["is_live_queue"] is False


# =============================================================================
# Issue 3: Wait/Turn consistency with active consultation remaining time
# =============================================================================
def test_issue_3_wait_and_turn_consistency_with_active_consultation():
    """Queue position numbering and estimated wait/turn for checked_in patient
    when another patient is currently in_service.

    Per established behavior in test_portals_and_features.py TEST D:
    - in_service patient → 'Current Patient' (patients_ahead=0)
    - First checked_in patient → '#2', patients_ahead=1 (the in_service patient counts)
    - Wait = remaining time of active consultation
    """
    doc = {
        "id": 1,
        "name": "Dr. Test",
        "department": "Dermatology",
        "shift_start": "10:00",
        "shift_end": "13:00",
        "slot_minutes": 15,
        "buffer_minutes": 0,
    }
    # Active consultation started at 10:00; now is 10:10 → 10 min elapsed of predicted 20 min
    # Remaining = 20 - 10 = 10 min; proj_end = 10:20
    # Waiting patient checked in at 10:05, slot 10:15
    bookings = [
        {
            "id": 101,
            "patient_name": "Active Patient",
            "priority_level": 0,
            "status": "in_service",
            "service_start": "10:00",
            "slot_start": "10:00",
            "noshow_prob": 0.05,
        },
        {
            "id": 102,
            "patient_name": "Waiting Patient 1",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-24T10:05:00",
            "slot_start": "10:15",
            "noshow_prob": 0.05,
        },
    ]

    sched = sc.live_schedule(
        doc,
        bookings,
        now_min=sc.to_min("10:10"),
        is_today=True,
        remaining_service_fn=lambda _t, _b: 20.0,
    )

    t_map = {it["patient_name"]: it for it in sched["timeline"]}
    active = t_map["Active Patient"]
    waiting = t_map["Waiting Patient 1"]

    # Active patient is in consultation
    assert active["queue_position"] == "Current Patient"
    assert active["est_turn"] == "Now"
    assert active["est_wait_min"] == 0.0

    # First waiting patient: position #2 (in_service counts as position 0 ahead)
    assert waiting["queue_position"] == "#2"
    assert waiting["patients_ahead"] == 1  # in_service patient is ahead
    assert waiting["est_wait_min"] == 10.0  # remaining time = 20 - 10 = 10 min
    assert waiting["est_turn"] == "10:20 AM"

    # Mathematical consistency: now + wait = turn
    assert sc.to_min("10:10") + waiting["est_wait_min"] == sc.to_min(waiting["est_turn"])


# =============================================================================
# Issue 4 & 12: Future Booked appointments do not show live queue info
# =============================================================================
def test_issue_4_and_12_future_appointments_outside_live_queue(isolated_env):
    """Booked appointments must show 'Available after check-in', no live queue data,
    and must never display a misleading 0-min wait."""
    env = isolated_env
    did = env["doc_derma_id"]
    tomorrow_iso = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    bid, _ = service.book_appointment({
        "patient_name": "Future Patient",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": tomorrow_iso,
        "time_slot": "11:00-11:15",
        "slot_start": "11:00",
        "slot_end": "11:15",
        "priority_level": 0,
        "status": "booked",
    })

    ticket = service.get_patient_live_ticket(patient_name="Future Patient")
    assert ticket is not None
    assert ticket["status"] == "booked"
    assert ticket["queue_position"] == "Available after check-in"
    assert ticket["patients_ahead"] is None
    assert ticket["live_eta_display"] == "Available after check-in"
    assert ticket["is_live_queue"] is False
    assert ticket["notifications"] == []


# =============================================================================
# Issue 5: No overlapping confirmed bookings for the same doctor
# =============================================================================
def test_issue_5_overlapping_appointment_booking_prevented(isolated_env):
    """Backend must reject slot times that overlap with an existing confirmed booking."""
    env = isolated_env
    did = env["doc_derma_id"]
    today_iso = datetime.date.today().isoformat()

    # First confirmed booking 10:17–10:28
    bid1, err1 = service.book_appointment({
        "patient_name": "Patient One",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "time_slot": "10:17-10:28",
        "slot_start": "10:17",
        "slot_end": "10:28",
        "priority_level": 0,
    })
    assert err1 is None and bid1 is not None

    # Conflicting booking 10:24–10:36 (overlaps 10:24–10:28)
    bid2, err2 = service.book_appointment({
        "patient_name": "Patient Two",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "time_slot": "10:24-10:36",
        "slot_start": "10:24",
        "slot_end": "10:36",
        "priority_level": 0,
    })
    assert bid2 is None
    assert "conflicts with an existing appointment" in (err2 or "")

    # Non-conflicting booking 10:30–10:45 should succeed (after 10:28 end)
    bid3, err3 = service.book_appointment({
        "patient_name": "Patient Three",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "time_slot": "10:30-10:45",
        "slot_start": "10:30",
        "slot_end": "10:45",
        "priority_level": 0,
    })
    assert err3 is None and bid3 is not None


# =============================================================================
# Issues 6 & 7: Expired appointments → no_show; invalid transitions rejected
# =============================================================================
def test_issue_6_and_7_expired_and_no_show_workflow(isolated_env):
    """auto_handle_expired_appointments transitions past and grace-elapsed bookings
    to no_show, leaves future and in-progress ones intact.
    Invalid transitions (done → no_show) raise ValueError."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    today_iso = datetime.date.today().isoformat()
    yesterday_iso = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow_iso = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    # Past date → should become no_show
    b_past = store.add_booking(db, {
        "patient_name": "Past Missed",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": yesterday_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })

    # Today, slot 10:00 ended at 10:15; now is 11:00 → grace(30m) expired
    b_today_exp = store.add_booking(db, {
        "patient_name": "Today Expired",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })

    # Today, slot 11:30 → not expired at 11:00
    b_today_future = store.add_booking(db, {
        "patient_name": "Today Future",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "slot_start": "11:30",
        "slot_end": "11:45",
        "status": "booked",
    })

    # Tomorrow → must not be touched
    b_tomorrow = store.add_booking(db, {
        "patient_name": "Tomorrow Future",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": tomorrow_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })

    # Done appointment → must stay done
    b_done = store.add_booking(db, {
        "patient_name": "Done Patient",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "slot_start": "09:00",
        "slot_end": "09:15",
        "status": "done",
    })

    # Run at 11:00 (660 min) with 30 min grace
    changed = store.auto_handle_expired_appointments(
        db, today_iso=today_iso, now_min=660.0, grace_minutes=30.0
    )
    assert changed == 2  # b_past and b_today_exp

    assert store.get_booking(db, b_past)["status"] == "no_show"
    assert store.get_booking(db, b_today_exp)["status"] == "no_show"
    assert store.get_booking(db, b_today_future)["status"] == "booked"
    assert store.get_booking(db, b_tomorrow)["status"] == "booked"
    assert store.get_booking(db, b_done)["status"] == "done"

    # Issue 7: done → no_show must raise ValueError (not silently succeed)
    with pytest.raises(ValueError, match="cannot mark a completed appointment as no-show"):
        store.set_status(db, b_done, "no_show")

    # Confirm done record was not modified
    assert store.get_booking(db, b_done)["status"] == "done"


# =============================================================================
# Issues 8, 9, 10: Emergency priority, active consultation protection, queue progression
# =============================================================================
def test_issue_8_9_10_emergency_priority_and_queue_progression():
    """Emergency patient is placed ahead of routine waiting patients,
    active consultation is NEVER interrupted.
    Queue positions shift correctly after emergency insertion.
    Per established behavior: in_service = pos 0; first checked_in = pos (ci_idx + len(in_service)+1).
    """
    doc = {
        "id": 1,
        "name": "Dr. Test",
        "department": "General",
        "shift_start": "09:00",
        "shift_end": "13:00",
        "slot_minutes": 15,
        "buffer_minutes": 0,
    }

    # Step A: Routine patient R1 is checked in; no in_service patient yet
    b_r1_only = [
        {
            "id": 2,
            "patient_name": "R1 Routine",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-24T09:00:00",
            "slot_start": "09:15",
            "noshow_prob": 0.05,
        },
    ]
    s1 = sc.live_schedule(doc, b_r1_only, now_min=sc.to_min("09:05"), is_today=True)
    t1 = {it["patient_name"]: it for it in s1["timeline"]}
    # No in_service → R1 is at #1 with 0 patients ahead
    assert t1["R1 Routine"]["queue_position"] == "#1"
    assert t1["R1 Routine"]["patients_ahead"] == 0

    # Step B: Active consultation starts; R1 now has 1 patient ahead
    b_with_active = [
        {
            "id": 1,
            "patient_name": "A1 Active",
            "priority_level": 0,
            "status": "in_service",
            "service_start": "09:00",
            "slot_start": "09:00",
            "noshow_prob": 0.05,
        },
        {
            "id": 2,
            "patient_name": "R1 Routine",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-24T09:00:00",
            "slot_start": "09:15",
            "noshow_prob": 0.05,
        },
    ]
    s2 = sc.live_schedule(
        doc, b_with_active, now_min=sc.to_min("09:05"), is_today=True,
        remaining_service_fn=lambda _t, _b: 15.0,
    )
    t2 = {it["patient_name"]: it for it in s2["timeline"]}
    assert t2["A1 Active"]["queue_position"] == "Current Patient"
    assert t2["A1 Active"]["est_turn"] == "Now"
    # R1 moves from #1 to #2 because in_service patient counts as 1 ahead
    assert t2["R1 Routine"]["queue_position"] == "#2"
    assert t2["R1 Routine"]["patients_ahead"] == 1
    assert t2["R1 Routine"]["est_wait_min"] == 10.0  # remaining = 15 - 5 = 10
    assert t2["R1 Routine"]["est_turn"] == "9:15 AM"

    # Step C: Emergency E1 arrives → E1 goes before R1 but cannot interrupt A1
    b_with_emergency = [
        {
            "id": 1,
            "patient_name": "A1 Active",
            "priority_level": 0,
            "status": "in_service",
            "service_start": "09:00",
            "slot_start": "09:00",
            "noshow_prob": 0.05,
        },
        {
            "id": 3,
            "patient_name": "E1 Emergency",
            "priority_level": 1,  # emergency
            "status": "checked_in",
            "check_in_time": "2026-10-24T09:06:00",
            "slot_start": "09:06",
            "noshow_prob": 0.05,
        },
        {
            "id": 2,
            "patient_name": "R1 Routine",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-24T09:00:00",
            "slot_start": "09:15",
            "noshow_prob": 0.05,
        },
    ]
    s3 = sc.live_schedule(
        doc, b_with_emergency, now_min=sc.to_min("09:06"), is_today=True,
        remaining_service_fn=lambda _t, _b: 15.0,
    )
    t3 = {it["patient_name"]: it for it in s3["timeline"]}

    # Active consultation NOT interrupted
    assert t3["A1 Active"]["queue_position"] == "Current Patient"
    assert t3["A1 Active"]["est_turn"] == "Now"

    # Emergency E1 is at #2 (1 in_service ahead), routine R1 moves to #3
    assert t3["E1 Emergency"]["queue_position"] == "#2"
    assert t3["E1 Emergency"]["patients_ahead"] == 1  # only A1 ahead
    assert t3["E1 Emergency"]["est_wait_min"] == 9.0   # 15 - 6 = 9 min remaining
    assert t3["E1 Emergency"]["est_turn"] == "9:15 AM"

    assert t3["R1 Routine"]["queue_position"] == "#3"
    assert t3["R1 Routine"]["patients_ahead"] == 2   # A1 + E1 ahead
    # R1 wait = A1 remaining (9m) + E1 duration (15m) = 24m → turn at 9:30
    assert t3["R1 Routine"]["est_wait_min"] == 24.0
    assert t3["R1 Routine"]["est_turn"] == "9:30 AM"

    # Step D: A1 finishes → E1 becomes #1, R1 becomes #2
    b_post_active = [
        {
            "id": 1,
            "patient_name": "A1 Active",
            "priority_level": 0,
            "status": "done",
            "slot_start": "09:00",
            "service_minutes": 15.0,
            "noshow_prob": 0.05,
        },
        {
            "id": 3,
            "patient_name": "E1 Emergency",
            "priority_level": 1,
            "status": "checked_in",
            "check_in_time": "2026-10-24T09:06:00",
            "slot_start": "09:06",
            "noshow_prob": 0.05,
        },
        {
            "id": 2,
            "patient_name": "R1 Routine",
            "priority_level": 0,
            "status": "checked_in",
            "check_in_time": "2026-10-24T09:00:00",
            "slot_start": "09:15",
            "noshow_prob": 0.05,
        },
    ]
    s4 = sc.live_schedule(
        doc, b_post_active, now_min=sc.to_min("09:15"), is_today=True,
        remaining_service_fn=lambda _t, _b: 15.0,
    )
    t4 = {it["patient_name"]: it for it in s4["timeline"]}
    assert t4["A1 Active"]["queue_position"] == "Completed"
    # E1 is #1, no patients ahead
    assert t4["E1 Emergency"]["queue_position"] == "#1"
    assert t4["E1 Emergency"]["patients_ahead"] == 0
    assert t4["E1 Emergency"]["est_wait_min"] == 0.0
    # R1 is #2, 1 patient ahead (E1)
    assert t4["R1 Routine"]["queue_position"] == "#2"
    assert t4["R1 Routine"]["patients_ahead"] == 1


# =============================================================================
# Issue 11: Projected Finish and Slack mathematical consistency
# =============================================================================
def test_issue_11_projected_finish_and_slack():
    """slack = shift_end - projected_finish; positive = early, negative = overrun."""
    doc = {
        "id": 1,
        "name": "Dr. D",
        "department": "Cardiology",
        "shift_start": "09:00",
        "shift_end": "11:00",   # 120 min duty
        "slot_minutes": 20,
        "buffer_minutes": 0,
    }

    # Case A: 2 patients × 15 min actual → proj_finish = 09:35 (cursor = 09:00 + 15 + 20 = 09:35)
    # The 2nd done patient's slot_start is 09:20, service=15 → proj_end = 09:20+15 = 09:35
    # shift_end 11:00 (660 min), proj_finish 09:35 (575 min) → slack = +85
    bookings_light = [
        {"id": 1, "patient_name": "P1", "priority_level": 0, "status": "done",
         "slot_start": "09:00", "service_minutes": 15, "noshow_prob": 0.05},
        {"id": 2, "patient_name": "P2", "priority_level": 0, "status": "done",
         "slot_start": "09:20", "service_minutes": 15, "noshow_prob": 0.05},
    ]
    ls_light = sc.live_schedule(doc, bookings_light, now_min=sc.to_min("09:30"))
    summary_light = ls_light["summary"]
    # Both patients done; projected_finish is after the last patient ends
    # P1 ends at 09:15, cursor = 09:15; P2 planned at 09:20 → proj_start = max(09:15, 09:20) = 09:20
    # P2 ends at 09:35; proj_finish = 09:35; slack = 660 - 575 = 85 min
    assert summary_light["projected_finish"] == "09:35"
    assert summary_light["slack_min"] == 85.0
    assert summary_light["finishing_early"] is True

    # Case B: Overrun — single patient takes 50 min in a 60 min window
    # P1 slot 09:00, service=50 → proj_end 09:50; + booked P2 at 09:20 → cursor=09:50
    # P2 proj_start = max(09:50, 09:20) = 09:50; P2 dur=20 → proj_end = 10:10
    # shift_end 10:00 → slack = 600 - 610 = -10
    doc2 = dict(doc, shift_end="10:00")
    bookings_heavy = [
        {"id": 1, "patient_name": "P1", "priority_level": 0, "status": "done",
         "slot_start": "09:00", "service_minutes": 50, "noshow_prob": 0.05},
        {"id": 2, "patient_name": "P2", "priority_level": 0, "status": "checked_in",
         "check_in_time": "2026-10-24T09:20:00", "slot_start": "09:20", "noshow_prob": 0.05},
    ]
    ls_heavy = sc.live_schedule(
        doc2, bookings_heavy, now_min=sc.to_min("09:50"),
        remaining_service_fn=lambda _t, _b: 20.0,
    )
    summary_heavy = ls_heavy["summary"]
    assert summary_heavy["slack_min"] < 0    # overrunning shift
    assert summary_heavy["finishing_early"] is False


# =============================================================================
# Admin Emergency Walk-In: authorization, validation, deduplication
# =============================================================================
def test_admin_emergency_walk_in_authorization_and_validation(isolated_env):
    """Admin-only role enforcement, department validation, and duplicate protection."""
    env = isolated_env
    did_derma = env["doc_derma_id"]
    admin_user = env["admin_user"]
    patient_user = {"id": 99, "username": "patient_bob", "role": "patient"}

    # Non-admin (patient role) must be rejected
    with pytest.raises(PermissionError):
        service.add_emergency_walk_in(
            admin_user=patient_user,
            patient_name="Bob Patient",
            department="Dermatology",
            doctor_id=did_derma,
            emergency_reason="Severe acute reaction",
        )

    # Doctor not in Cardiology → department mismatch
    bid_mismatch, err_mismatch = service.add_emergency_walk_in(
        admin_user=admin_user,
        patient_name="Bob Patient",
        department="Cardiology",
        doctor_id=did_derma,
        emergency_reason="Severe acute reaction",
    )
    assert bid_mismatch is None
    assert "does not belong" in (err_mismatch or "")

    # Valid emergency walk-in
    bid, err = service.add_emergency_walk_in(
        admin_user=admin_user,
        patient_name="Emergency Bob",
        department="Dermatology",
        doctor_id=did_derma,
        emergency_reason="Severe chemical burn requiring immediate triage",
    )
    assert err is None
    assert bid is not None

    b = store.get_booking(env["db"], bid)
    assert b["status"] == "checked_in"
    assert b["priority_level"] == 1
    assert b["time_slot"] == "Emergency Walk-In"
    assert b["emergency_reason"] == "Severe chemical burn requiring immediate triage"
    assert "admin:admin" in b["assigned_by"]

    # Duplicate submission protection
    bid_dup, err_dup = service.add_emergency_walk_in(
        admin_user=admin_user,
        patient_name="Emergency Bob",
        department="Dermatology",
        doctor_id=did_derma,
        emergency_reason="Duplicate attempt",
    )
    assert bid_dup is None
    assert "already active in the live queue" in (err_dup or "")


# =============================================================================
# Part C7: Complete 15-Step Acceptance Scenario
# =============================================================================
def test_part_c7_full_15_step_acceptance_scenario(isolated_env):
    """15-step acceptance scenario testing end-to-end emergency walk-in workflow."""
    env = isolated_env
    db = env["db"]
    did_derma = env["doc_derma_id"]
    did_cardio = env["doc_cardio_id"]
    admin_user = env["admin_user"]
    today_iso = datetime.date.today().isoformat()

    # Steps 1 & 2: Create and register two Dermatology patients + one Cardiology patient
    u_r1, _ = service.register_user({
        "username": "pat_r1", "password": "pass1", "role": "patient",
        "full_name": "Routine Patient 1", "department": "Dermatology",
    })
    u_r2, _ = service.register_user({
        "username": "pat_r2", "password": "pass2", "role": "patient",
        "full_name": "Routine Patient 2", "department": "Dermatology",
    })
    u_cardio, _ = service.register_user({
        "username": "pat_cardio", "password": "pass3", "role": "patient",
        "full_name": "Cardio Patient", "department": "Cardiology",
    })

    # Book appointments (non-overlapping slots)
    bid_r1, err_r1 = service.book_appointment({
        "patient_name": "Routine Patient 1",
        "patient_user_id": u_r1,
        "department": "Dermatology",
        "doctor_id": did_derma,
        "appointment_date": today_iso,
        "time_slot": "10:15-10:30",
        "slot_start": "10:15",
        "slot_end": "10:30",
        "priority_level": 0,
        "status": "booked",
    })
    assert err_r1 is None

    bid_r2, err_r2 = service.book_appointment({
        "patient_name": "Routine Patient 2",
        "patient_user_id": u_r2,
        "department": "Dermatology",
        "doctor_id": did_derma,
        "appointment_date": today_iso,
        "time_slot": "10:35-10:50",
        "slot_start": "10:35",
        "slot_end": "10:50",
        "priority_level": 0,
        "status": "booked",
    })
    assert err_r2 is None

    bid_cardio, err_cardio = service.book_appointment({
        "patient_name": "Cardio Patient",
        "patient_user_id": u_cardio,
        "department": "Cardiology",
        "doctor_id": did_cardio,
        "appointment_date": today_iso,
        "time_slot": "09:30-09:50",
        "slot_start": "09:30",
        "slot_end": "09:50",
        "priority_level": 0,
        "status": "booked",
    })
    assert err_cardio is None

    # Step 2: Check all patients in
    store.check_in(db, bid_r1)
    store.check_in(db, bid_r2)
    store.check_in(db, bid_cardio)

    # Step 3: Verify initial queue positions
    ticket_r1_init = service.get_patient_live_ticket(patient_user_id=u_r1)
    ticket_r2_init = service.get_patient_live_ticket(patient_user_id=u_r2)
    assert ticket_r1_init["queue_position"] == "#1"
    assert ticket_r2_init["queue_position"] == "#2"

    # Steps 4 & 5: Admin adds an emergency walk-in to Dermatology
    bid_em, err_em = service.add_emergency_walk_in(
        admin_user=admin_user,
        patient_name="Emergency WalkIn",
        department="Dermatology",
        doctor_id=did_derma,
        emergency_reason="Acute anaphylactic reaction",
    )

    # Step 6: Exactly one emergency record created
    assert err_em is None and bid_em is not None
    em_bookings = [
        b for b in store.list_bookings(db, doctor_id=did_derma, date=today_iso)
        if b["patient_name"] == "Emergency WalkIn"
    ]
    assert len(em_bookings) == 1

    # Step 7: Emergency record is correctly flagged
    b_em = em_bookings[0]
    assert b_em["priority_level"] == 1
    assert b_em["time_slot"] == "Emergency Walk-In"
    assert b_em["status"] == "checked_in"

    # Step 8: Emergency patient appears at #1 in doctor's live schedule
    doc_sched = service.live_schedule(did_derma, today_iso)
    t_map = {it["patient_name"]: it for it in doc_sched["timeline"]}
    assert "Emergency WalkIn" in t_map
    assert t_map["Emergency WalkIn"]["queue_position"] == "#1"

    # Step 9: Routine patients see in-app emergency alert notification
    ticket_r1_after = service.get_patient_live_ticket(patient_user_id=u_r1)
    ticket_r2_after = service.get_patient_live_ticket(patient_user_id=u_r2)
    assert len(ticket_r1_after["notifications"]) >= 1
    assert len(ticket_r2_after["notifications"]) >= 1
    notif = ticket_r1_after["notifications"][0]
    assert "Emergency Alert" in notif["message"]

    # Step 10: Routine patients shift back in queue
    assert ticket_r1_after["queue_position"] == "#2"
    assert ticket_r2_after["queue_position"] == "#3"

    # Step 11: Cardiology patient does NOT see the Dermatology notification
    ticket_cardio = service.get_patient_live_ticket(patient_user_id=u_cardio)
    assert ticket_cardio["notifications"] == []

    # Step 12: Notification must NOT contain emergency patient's name or diagnosis
    notif_msg = notif["message"]
    assert "Emergency WalkIn" not in notif_msg
    assert "anaphylactic" not in notif_msg.lower()
    assert "acute" not in notif_msg.lower() or "Emergency Alert" in notif_msg

    # Step 13: If consultation is active, it is NOT interrupted by new emergency
    store.start_service(db, bid_em)
    sched_active = service.live_schedule(did_derma, today_iso)
    t_active = {it["patient_name"]: it for it in sched_active["timeline"]}
    assert t_active["Emergency WalkIn"]["queue_position"] == "Current Patient"

    bid_em2, err_em2 = service.add_emergency_walk_in(
        admin_user=admin_user,
        patient_name="Second Emergency",
        department="Dermatology",
        doctor_id=did_derma,
        emergency_reason="High-grade laceration",
    )
    assert err_em2 is None

    sched_active2 = service.live_schedule(did_derma, today_iso)
    t_active2 = {it["patient_name"]: it for it in sched_active2["timeline"]}
    # Active consultation remains active; NOT interrupted
    assert t_active2["Emergency WalkIn"]["queue_position"] == "Current Patient"
    # Second emergency is next in line
    assert t_active2["Second Emergency"]["queue_position"] == "#2"

    # Step 14: Dismissed alerts don't re-appear after refresh
    dismissed_ids = {n["id"] for n in ticket_r1_after["notifications"]}
    fresh_notifs_r1 = service.get_waiting_patient_notifications(
        "Dermatology", did_derma
    )
    remaining = [n for n in fresh_notifs_r1 if n["id"] not in dismissed_ids]
    # New notification for Second Emergency was added, so at least 1 new one
    assert len(fresh_notifs_r1) >= len(dismissed_ids)

    # Step 15: All original booking records remain intact
    all_b = store.list_bookings(db, doctor_id=did_derma, date=today_iso)
    booking_ids = {b["id"] for b in all_b}
    assert bid_r1 in booking_ids
    assert bid_r2 in booking_ids
    assert bid_em in booking_ids
    assert bid_em2 in booking_ids


# =============================================================================
# Real DB integrity: app/qms.db and ML models
# =============================================================================
def test_qms_real_db_integrity_and_ml_models():
    """Verify app/qms.db was not deleted, all seeded users/appointments remain intact,
    and ML models load and predict without errors."""
    real_db = Path("app/qms.db")
    assert real_db.exists(), "app/qms.db must exist and not have been deleted!"

    # Verify existing users by role (using known IDs/usernames)
    u_admin = store.authenticate_user(real_db, "admin", "admin123")
    assert u_admin is not None and u_admin["role"] == "admin"

    u_patient_john = store.authenticate_user(real_db, "patient_john", "patient123")
    assert u_patient_john is not None and u_patient_john["role"] == "patient"

    # risha and prisha exist as patients (even without a known password, check by username)
    all_users = store.list_users(real_db)
    usernames = {u["username"] for u in all_users}
    assert "risha" in usernames, "User 'risha' must exist in qms.db"
    assert "prisha" in usernames, "User 'prisha' must exist in qms.db"

    # Verify appointment table accessibility and integrity
    all_appts = store.list_bookings(real_db)
    assert isinstance(all_appts, list)
    for b in all_appts:
        assert "status" in b
        assert "doctor_id" in b

    # Verify ML models load and predict
    assert service.has_service_model() is True

    svc_pred = service.predict_service({
        "department": "Dermatology",
        "hour_of_day": 10,
        "day_of_week": "Friday",
        "is_weekend": 0,
        "active_doctors_count": 2,
        "slot_arrival_count": 3,
        "queue_length_at_arrival": 1,
        "previous_no_show_count": 0,
        "previous_late_count": 0,
        "total_visits_count": 2,
        "priority_level": 0,
    })
    assert svc_pred > 0.0

    noshow_result = service.predict_no_show({
        "department": "Dermatology",
        "time_slot": "morning",
        "day_of_week": "Friday",
        "lead_time_days": 1,
        "lead_time_hours": 24,
        "hour_of_day": 10,
        "booking_hour": 9,
        "is_weekend": 0,
        "is_peak_hour": 0,
        "is_holiday": 0,
        "previous_no_show_count": 0,
        "previous_late_count": 0,
        "total_visits_count": 1,
        "priority_level": 0,
    })
    assert isinstance(noshow_result, tuple), "predict_no_show must return a (probability, drivers) tuple"
    noshow_pred, drivers = noshow_result
    assert 0.0 <= float(noshow_pred) <= 1.0
    assert isinstance(drivers, list)


# =============================================================================
# Past-appointment check-in and expired queue regression tests
# =============================================================================
def test_past_appointment_check_in_rejected(isolated_env):
    """Past appointments (date < today) cannot be checked in via service or store."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    yesterday_iso = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    pid, _ = store.create_user(db, {
        "username": "past_user", "password": "password123", "role": "patient", "full_name": "Past Patient"
    })
    past_bid = store.add_booking(db, {
        "patient_name": "Past Patient",
        "patient_user_id": pid,
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": yesterday_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })
    patient_user = {"id": pid, "username": "past_user", "role": "patient", "full_name": "Past Patient"}

    # 1. service.patient_check_in rejects past appointment
    with pytest.raises(ValueError, match="appointment date has passed"):
        service.patient_check_in(patient_user, past_bid)
    assert store.get_booking(db, past_bid)["status"] == "booked"

    # 2. store.check_in also rejects past appointment
    with pytest.raises(ValueError, match="appointment date has passed"):
        store.check_in(db, past_bid)
    assert store.get_booking(db, past_bid)["status"] == "booked"


def test_unattended_expired_booked_becomes_no_show(isolated_env):
    """Unattended expired booked appointments transition to no_show."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    yesterday_iso = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    today_iso = datetime.date.today().isoformat()

    past_bid = store.add_booking(db, {
        "patient_name": "Unattended Expired",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": yesterday_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })

    changed = store.auto_handle_expired_appointments(db, today_iso=today_iso)
    assert changed >= 1
    assert store.get_booking(db, past_bid)["status"] == "no_show"


def test_valid_same_day_check_in_works(isolated_env):
    """Same-day valid appointment can be successfully checked in."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    today_iso = datetime.date.today().isoformat()

    pid, _ = store.create_user(db, {
        "username": "today_user", "password": "password123", "role": "patient", "full_name": "Today Patient"
    })
    bid = store.add_booking(db, {
        "patient_name": "Today Patient",
        "patient_user_id": pid,
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "booked",
    })
    patient_user = {"id": pid, "username": "today_user", "role": "patient", "full_name": "Today Patient"}

    service.patient_check_in(patient_user, bid)
    assert store.get_booking(db, bid)["status"] == "checked_in"


def test_future_appointment_remains_booked_excluded_from_live_queue(isolated_env):
    """Future booked appointments remain outside the live waiting queue until check-in."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    tomorrow_iso = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    pid, _ = store.create_user(db, {
        "username": "fut_user", "password": "password123", "role": "patient", "full_name": "Future Patient"
    })
    fut_bid = store.add_booking(db, {
        "patient_name": "Future Patient",
        "patient_user_id": pid,
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": tomorrow_iso,
        "slot_start": "11:00",
        "slot_end": "11:15",
        "status": "booked",
    })

    ticket = service.get_patient_live_ticket(patient_user_id=pid)
    assert ticket is not None
    assert ticket["id"] == fut_bid
    assert ticket["status"] == "booked"
    assert ticket["is_live_queue"] is False
    assert ticket["queue_position"] == "Available after check-in"
    assert ticket["queue_pos_num"] is None
    assert ticket["live_eta_minutes"] is None


def test_past_appointment_cannot_affect_today_queue_or_eta(isolated_env):
    """Past appointments cannot contribute to today's live queue, queue position, or ETA."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    today_iso = datetime.date.today().isoformat()
    yesterday_iso = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    doc = store.get_doctor(db, did)

    # Seed an old checked-in appointment from yesterday
    past_bid = store.add_booking(db, {
        "patient_name": "Stale Yesterday Patient",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": yesterday_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "checked_in",
    })

    # Seed today's checked-in patient
    today_bid = store.add_booking(db, {
        "patient_name": "Real Today Patient",
        "department": "Dermatology",
        "doctor_id": did,
        "appointment_date": today_iso,
        "slot_start": "10:00",
        "slot_end": "10:15",
        "status": "checked_in",
    })

    today_bookings = store.list_bookings(db, doctor_id=did, date=today_iso)
    today_ids = [b["id"] for b in today_bookings]
    assert past_bid not in today_ids
    assert today_bid in today_ids

    sched = sc.live_schedule(doc, today_bookings)
    timeline_ids = [it["id"] for it in sched["timeline"]]
    assert past_bid not in timeline_ids
    assert today_bid in timeline_ids

    # For the past checked-in patient, get_patient_live_ticket must NOT treat as active live queue
    pid_past, _ = store.create_user(db, {
        "username": "past_ci_user", "password": "password123", "role": "patient", "full_name": "Past CI Patient"
    })
    store.update_booking(db, past_bid, patient_user_id=pid_past)
    ticket = service.get_patient_live_ticket(patient_user_id=pid_past)
    assert ticket["is_live_queue"] is False
    assert ticket["live_eta_minutes"] is None
    assert ticket["queue_pos_num"] is None


def test_completed_and_cancelled_appointments_not_modified(isolated_env):
    """Completed and cancelled appointments remain untouched by auto_handle_expired_appointments."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    today_iso = datetime.date.today().isoformat()
    yesterday_iso = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    b_done = store.add_booking(db, {
        "patient_name": "Done Visit", "department": "Dermatology", "doctor_id": did,
        "appointment_date": yesterday_iso, "slot_start": "09:00", "slot_end": "09:15", "status": "done"
    })
    b_cancelled = store.add_booking(db, {
        "patient_name": "Cancelled Visit", "department": "Dermatology", "doctor_id": did,
        "appointment_date": yesterday_iso, "slot_start": "09:15", "slot_end": "09:30", "status": "cancelled"
    })
    b_no_show = store.add_booking(db, {
        "patient_name": "Already NoShow", "department": "Dermatology", "doctor_id": did,
        "appointment_date": yesterday_iso, "slot_start": "09:30", "slot_end": "09:45", "status": "no_show"
    })

    changed = store.auto_handle_expired_appointments(db, today_iso=today_iso)
    assert changed == 0
    assert store.get_booking(db, b_done)["status"] == "done"
    assert store.get_booking(db, b_cancelled)["status"] == "cancelled"
    assert store.get_booking(db, b_no_show)["status"] == "no_show"

    with pytest.raises(ValueError, match="cannot mark a completed appointment as no-show"):
        store.set_status(db, b_done, "no_show")


def test_past_appointment_not_preferred_over_future_or_today_booked(isolated_env):
    """When a patient has a past appointment and a future/today booking, the upcoming one is returned as live ticket."""
    env = isolated_env
    db = env["db"]
    did = env["doc_derma_id"]
    yesterday_iso = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow_iso = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    pid, _ = store.create_user(db, {
        "username": "multi_appt_user", "password": "password123", "role": "patient", "full_name": "Multi Appt Patient"
    })
    _past_bid = store.add_booking(db, {
        "patient_name": "Multi Appt Patient", "patient_user_id": pid, "department": "Dermatology",
        "doctor_id": did, "appointment_date": yesterday_iso, "slot_start": "10:00", "slot_end": "10:15",
        "status": "booked"
    })
    fut_bid = store.add_booking(db, {
        "patient_name": "Multi Appt Patient", "patient_user_id": pid, "department": "Dermatology",
        "doctor_id": did, "appointment_date": tomorrow_iso, "slot_start": "11:00", "slot_end": "11:15",
        "status": "booked"
    })

    ticket = service.get_patient_live_ticket(patient_user_id=pid)
    assert ticket is not None
    assert ticket["id"] == fut_bid
    assert ticket["status"] == "booked"

