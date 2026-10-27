"""Verification tests for QMS modifications."""

import datetime
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

import service
import store


def test_auth_and_passwords():
    db = service.db_path()
    service.seed_default_system()
    store.seed_default_users_and_doctors(db)

    # 1. Test Patient Auth
    p_user, p_err = service.authenticate_user("patient_john", "patient123", expected_role="patient")
    assert p_user is not None, f"Patient auth failed: {p_err}"
    assert p_user["role"] == "patient"

    # 2. Test Doctor Auth
    d_user, d_err = service.authenticate_user("dr_sharma", "doctor123", expected_role="doctor")
    assert d_user is not None, f"Doctor auth failed: {d_err}"
    assert d_user["role"] == "doctor"
    assert d_user.get("doctor_id") is not None

    # 3. Test Admin Auth
    a_user, a_err = service.authenticate_user("admin", "admin123", expected_role="admin")
    assert a_user is not None, f"Admin auth failed: {a_err}"
    assert a_user["role"] == "admin"

    # 4. Test Cross-role Rejection
    bad_doc, bad_err1 = service.authenticate_user("patient_john", "patient123", expected_role="doctor")
    assert bad_doc is None
    assert "registered as **patient**" in bad_err1

    bad_admin, bad_err2 = service.authenticate_user("dr_sharma", "doctor123", expected_role="admin")
    assert bad_admin is None
    assert "registered as **doctor**" in bad_err2

    print("[OK] Auth and role enforcement verified.")


def test_all_hospital_doctors_and_credentials():
    db = service.db_path()
    all_docs = service.list_all_hospital_doctors()
    assert len(all_docs) >= 8, f"Expected at least 8 doctors, got {len(all_docs)}"

    for doc in all_docs:
        assert doc.get("name"), "Doctor must have a name"
        assert doc.get("department"), "Doctor must have a department"
        assert doc.get("username"), f"Doctor {doc['name']} must have a username"
        assert doc.get("login_password"), f"Doctor {doc['name']} must have a login_password"
        assert doc.get("active") in (0, 1, True, False), f"Doctor {doc['name']} must have active status"

        # Verify credentials can authenticate
        u, err = service.authenticate_user(doc["username"], doc["login_password"], expected_role="doctor")
        assert u is not None, f"Failed to authenticate doctor {doc['username']}: {err}"
        assert u["doctor_id"] == doc["id"], f"Linked doctor_id mismatch for {doc['username']}"

    print(f"[OK] All {len(all_docs)} hospital doctors have valid login credentials and linked accounts.")


def test_inactive_doctor_visibility():
    db = service.db_path()
    all_docs = service.list_all_hospital_doctors()
    test_doc = all_docs[0]
    doc_id = test_doc["id"]

    try:
        # Set doctor to inactive
        store.set_doctor_status(db, doc_id, False)

        # Inactive doctor MUST still appear in all hospital doctors
        docs_after = service.list_all_hospital_doctors()
        found_in_all = [d for d in docs_after if d["id"] == doc_id]
        assert len(found_in_all) == 1, "Inactive doctor disappeared from hospital doctors list!"
        assert found_in_all[0]["active"] == 0, "Doctor should be inactive"
        assert found_in_all[0]["login_password"] is not None

        # Inactive doctor MUST NOT appear in active-only list (used by dashboard)
        active_docs = store.list_doctors(db, active_only=True)
        assert not any(d["id"] == doc_id for d in active_docs), "Inactive doctor appeared in active doctors list!"

    finally:
        # Restore active status
        store.set_doctor_status(db, doc_id, True)

    restored = [d for d in service.list_all_hospital_doctors() if d["id"] == doc_id][0]
    assert restored["active"] == 1, "Doctor failed to restore active status"
    print("[OK] Inactive doctor visibility and dashboard exclusion verified.")


def test_dashboard_active_doctors_patient_queue():
    db = service.db_path()
    today = datetime.date.today().isoformat()
    target_date = today if store.list_bookings(db, date=today) else "2026-10-03"

    active_docs = store.list_doctors(db, active_only=True)
    assert len(active_docs) > 0, "Should have active doctors"

    lifecycle_display = {
        "booked": "Booked",
        "checked_in": "Checked In",
        "in_service": "In Service",
        "done": "Completed",
        "no_show": "No Show",
        "cancelled": "Cancelled",
    }

    found_patient_with_status = False

    for doc in active_docs:
        appts = store.list_bookings(db, doctor_id=doc["id"], date=target_date)
        if not appts and "Ananya" in doc.get("name", ""):
            appts = store.list_bookings(db, doctor_id=doc["id"], date="2026-10-03")
        assert isinstance(appts, list)

        if doc["department"] == "Dermatology" and "Ananya" in doc["name"]:
            # Dr. Ananya Rao should have the existing patients in the DB
            assert len(appts) >= 2, f"Expected at least 2 patients for Dr. Ananya Rao, got {len(appts)}"
            patient_names = [a["patient_name"].lower() for a in appts]
            assert "risha" in patient_names, "Expected patient risha for Dr. Ananya Rao"
            assert "prisha" in patient_names, "Expected patient prisha for Dr. Ananya Rao"

            for a in appts:
                status_raw = a["status"]
                assert status_raw in lifecycle_display, f"Unknown status: {status_raw}"
                display_status = lifecycle_display[status_raw]
                assert display_status in ("Booked", "Checked In", "In Service", "Completed", "No Show", "Cancelled")
                found_patient_with_status = True

    assert found_patient_with_status, "Did not verify patient statuses"
    print("[OK] Dashboard active doctors and live patient queue verified against SQLite DB.")


if __name__ == "__main__":
    test_auth_and_passwords()
    test_all_hospital_doctors_and_credentials()
    test_inactive_doctor_visibility()
    test_dashboard_active_doctors_patient_queue()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
