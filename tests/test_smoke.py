"""End-to-end smoke tests for the QMS.

Runs with either pytest *or* plain ``python tests/test_smoke.py`` (no pytest
required). Exercises every layer on small synthetic inputs so it is fast and has
no dependency on scikit-learn, Streamlit, or a network connection.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]


def _toy_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.to_datetime("2026-04-01") + pd.to_timedelta(rng.integers(0, 30, n), "D")
    depts = rng.choice(["Cardiology", "Dentistry", "General"], n)
    slots = rng.choice(["morning", "afternoon", "evening"], n)
    prev_ns = rng.integers(0, 4, n)
    prob = 0.08 + 0.10 * prev_ns
    no_show = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "appointment_id": np.arange(n),
        "patient_id": rng.integers(0, 80, n),
        "appointment_date": dates,
        "booking_timestamp": dates - pd.to_timedelta(rng.integers(1, 240, n), "h"),
        "department": depts, "time_slot": slots,
        "day_of_week": pd.Categorical(dates.day_name()),
        "queue_id": rng.choice(["Q1", "Q2"], n),
        "priority_level": rng.choice([0, 1], n, p=[0.9, 0.1]),
        "lead_time_hours": rng.integers(1, 480, n).astype(float),
        "lead_time_days": rng.integers(0, 20, n).astype(float),
        "hour_of_day": rng.integers(8, 18, n), "booking_hour": rng.integers(8, 18, n),
        "is_weekend": rng.integers(0, 2, n), "is_peak_hour": rng.integers(0, 2, n),
        "is_holiday": 0,
        "previous_no_show_count": prev_ns, "previous_late_count": rng.integers(0, 3, n),
        "total_visits_count": rng.integers(1, 20, n),
        "active_doctors_count": rng.integers(1, 4, n),
        "slot_arrival_count": rng.integers(1, 15, n),
        "queue_length_at_arrival": rng.integers(0, 12, n),
        "buffer_window_minutes": 10,
        "no_show": no_show,
        "arrival_delay": rng.normal(0, 10, n).round(1),
        "service_time": rng.uniform(5, 25, n).round(1),
        "waiting_time": rng.uniform(0, 40, n).round(1),
        "check_in_time": dates + pd.to_timedelta(rng.integers(480, 1080, n), "m"),
    })


def test_preprocess_and_split():
    from qms.data import preprocess
    df = _toy_frame()
    X, y, rows = preprocess.get_task_frame(df, "no_show")
    assert len(X) == len(y) == len(rows)
    cats = preprocess.categorical_for("no_show")
    cols = preprocess.fit_encoder(X, cats)
    Xt = preprocess.transform(X, cols, cats)
    assert list(Xt.columns) == cols and Xt.isna().sum().sum() == 0
    mask = preprocess.temporal_split(rows, 0.2)
    assert 0 < mask.sum() < len(rows)


def test_arrival_delay_filters_noshows_and_lags():
    from qms.data import preprocess
    df = pd.DataFrame({
        "appointment_id": [1, 2, 3],
        "patient_id": ["P1", "P1", "P1"],
        "appointment_date": pd.to_datetime(["2026-04-01", "2026-04-08", "2026-04-15"]),
        "department": ["ENT", "ENT", "ENT"],
        "hour_of_day": [9, 9, 9],
        "no_show": [0, 1, 0],
        "arrival_delay": [12.0, 40.0, 6.0],
        "lead_time_hours": [24, 24, 24],
        "lead_time_days": [1, 1, 1],
        "booking_hour": [8, 8, 8],
        "is_weekend": [0, 0, 0],
        "is_peak_hour": [1, 1, 1],
        "is_holiday": [0, 0, 0],
        "priority_level": [0, 0, 0],
        "previous_no_show_count": [0, 0, 1],
        "previous_late_count": [0, 1, 1],
        "total_visits_count": [1, 2, 3],
        "active_doctors_count": [1, 1, 1],
        "slot_arrival_count": [2, 2, 2],
        "time_slot": ["morning", "morning", "morning"],
        "day_of_week": ["Wednesday", "Wednesday", "Wednesday"],
    })
    X, y, rows = preprocess.get_task_frame(df, "arrival_delay")
    assert len(y) == 2
    assert list(y) == [12.0, 6.0]
    assert "dept_mean_delay" in X.columns
    # first completed visit has no history; second completed sees only the 12 min delay
    assert float(X.iloc[0]["dept_mean_delay"]) == 0.0
    assert float(X.iloc[1]["dept_mean_delay"]) == 12.0
    assert abs(float(X.iloc[0]["hour_sin"])) > 0


def test_models_train_predict_and_bundle():
    from qms.data import preprocess
    from qms.models import estimators
    df = _toy_frame()
    for task in ["no_show", "arrival_delay", "waiting_time", "service_time"]:
        numeric, categorical, target, _ = preprocess.TASKS[task]
        X, y, rows = preprocess.get_task_frame(df, task)
        cols = preprocess.fit_encoder(X, categorical)
        Xm = preprocess.transform(X, cols, categorical).values
        if task == "no_show":
            est = estimators.make_classifier(0).fit(Xm, y.values)
            proba = est.predict_proba(Xm)
            assert proba.shape == (len(y), 2)
            kind = "classification"
        else:
            est = estimators.make_regressor(0).fit(Xm, y.values)
            assert est.predict(Xm).shape == (len(y),)
            kind = "regression"
        bundle = estimators.ModelBundle(
            task=task, kind=kind, target_name=target, estimator=est,
            feature_columns=cols, numeric=numeric, categorical=categorical,
            backend=estimators.backend_name())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"{task}.pkl"
            bundle.save(p)
            reloaded = estimators.ModelBundle.load(p)
        out = reloaded.predict(X)
        assert len(out) == len(y)


def test_objectives_buffer_scheduler_simulation():
    from qms.optimization.objectives import jain_fairness, composite_score
    from qms.optimization.buffer import DynamicBuffer
    from qms.optimization.scheduler import get_policy, POLICIES
    from qms.simulation.simulate import simulate

    assert abs(jain_fairness([5, 5, 5, 5]) - 1.0) < 1e-9
    assert jain_fairness([0, 0, 100]) < 0.6

    rec = DynamicBuffer().recommend([0.2, 0.3, 0.5, 0.4, 0.1])
    assert rec["n_booked"] == 5 and rec["recommended_overbook"] <= rec["max_overbook_allowed"]

    patients = [{"id": i, "ready": float(i * 2), "service": 10.0,
                 "priority": 1 if i == 3 else 0} for i in range(6)]
    metric_rows = {}
    for name in POLICIES:
        sim = simulate(patients, n_servers=2, policy=get_policy(name))
        assert len(sim["waits"]) == len(patients)
        from qms.optimization.objectives import summarize
        metric_rows[name] = summarize(sim)
    weights = {"waiting": 0.4, "utilization": 0.2, "fairness": 0.2, "emergency": 0.2}
    scores = composite_score(metric_rows, weights)
    assert set(scores) == set(POLICIES)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_genai_offline_fallback():
    from unittest.mock import patch
    import os

    orig_key = os.environ.get("GEMINI_API_KEY")
    os.environ.pop("GEMINI_API_KEY", None)
    try:
        with patch("qms.genai.explainer._parse_env_file", return_value=None):
            from qms.genai.explainer import GenAIExplainer
            exp = GenAIExplainer()          # no key -> offline template
            assert exp.mode == "offline-template"
            txt = exp.explain_no_show({"probability": 0.62, "drivers": ["prior no-shows"]})
            assert "62%" in txt and len(txt) > 20
            assert "min" in exp.explain_wait_estimate({"predicted_wait": 15, "queue_length": 4})
            arr = exp.explain_arrival({"delay_min": 8, "slot_start": "10:00"})
            assert "10:00" in arr and "8" in arr
            brief = exp.explain_live_day({
                "doctor_name": "Dr Rao", "date": "2026-04-01",
                "booked_count": 4, "done_count": 1, "new_open_slots": 2,
                "safe_overbook": 1, "projected_finish": "12:40", "slack_min": 20,
                "shift_end": "13:00", "expected_no_shows": 0.4,
            })
            assert "Dr Rao" in brief and "12:40" in brief
            ans = exp.answer("How many booked?", {"today": "2026-04-01", "booked": 3})
            assert "booked" in ans.lower() or "3" in ans
    finally:
        if orig_key is not None:
            os.environ["GEMINI_API_KEY"] = orig_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)



def test_store_lifecycle():
    import store
    import datetime
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        today_iso = datetime.date.today().isoformat()
        bid = store.add_booking(db, {
            "patient_name": "A", "department": "General",
            "appointment_date": today_iso, "time_slot": "morning",
            "day_of_week": "Wednesday", "priority_level": 1,
            "lead_time_hours": 24, "previous_no_show_count": 0, "noshow_prob": 0.2})
        store.check_in(db, bid, predicted_wait=8.0)
        store.set_status(db, bid, "in_service")
        rows = store.list_bookings(db)
        assert rows[0]["status"] == "in_service" and rows[0]["predicted_wait"] == 8.0
        assert store.counts_by_status(db)["in_service"] == 1

        # Expired check-in must be rejected
        past_bid = store.add_booking(db, {
            "patient_name": "Past", "department": "General",
            "appointment_date": "2020-01-01", "time_slot": "morning",
        })
        try:
            store.check_in(db, past_bid)
            raise AssertionError("Expired check-in should have raised ValueError")
        except ValueError as exc:
            assert "appointment date has passed" in str(exc)


def test_patient_memory():
    import store
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        store.add_booking(db, {
            "patient_name": "Alice", "department": "General",
            "appointment_date": "2026-08-01", "time_slot": "morning",
            "day_of_week": "Saturday", "priority_level": 0,
            "lead_time_hours": 24, "previous_no_show_count": 0, "noshow_prob": 0.1})
        bid = store.add_booking(db, {
            "patient_name": "alice", "department": "Cardiology",
            "appointment_date": "2026-08-15", "time_slot": "afternoon",
            "day_of_week": "Saturday", "priority_level": 0,
            "lead_time_hours": 24, "previous_no_show_count": 0, "noshow_prob": 0.4})
        store.set_status(db, bid, "no_show")
        known = store.list_known_patients(db)
        assert len(known) == 1 and known[0]["visits"] == 2 and known[0]["no_shows"] == 1
        prof = store.patient_profile(db, "ALICE")
        assert prof and prof["no_shows"] == 1 and prof["last_department"] == "Cardiology"


def test_auth_and_user_roles():
    import store
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "auth.db"
        store.seed_default_users_and_doctors(db)
        
        # Test seeded accounts
        u_patient = store.authenticate_user(db, "patient_john", "patient123")
        assert u_patient is not None and u_patient["role"] == "patient"
        
        u_doc = store.authenticate_user(db, "dr_sharma", "doctor123")
        assert u_doc is not None and u_doc["role"] == "doctor"
        
        u_admin = store.authenticate_user(db, "admin", "admin123")
        assert u_admin is not None and u_admin["role"] == "admin"
        
        # Test invalid credentials
        assert store.authenticate_user(db, "admin", "wrong_pass") is None
        assert store.authenticate_user(db, "non_existent_user", "1234") is None
        
        # Test new registration
        uid, err = store.create_user(db, {
            "username": "new_patient_sam",
            "password": "secure_pass_456",
            "role": "patient",
            "full_name": "Sam Taylor",
            "department": "Cardiology"
        })
        assert uid is not None and err is None
        
        # Authenticate newly created user
        u_new = store.authenticate_user(db, "new_patient_sam", "secure_pass_456")
        assert u_new is not None and u_new["full_name"] == "Sam Taylor"



def test_scheduling_plan_and_available():
    import scheduling as sc
    slots = sc.plan_slots("09:00", "13:00", 15, 5)   # step 20 over 240 min
    assert len(slots) == 12
    assert slots[0]["start"] == "09:00" and slots[1]["start"] == "09:20"
    doc = {"id": 1, "shift_start": "09:00", "shift_end": "13:00",
           "slot_minutes": 15, "buffer_minutes": 5}
    booked = [{"id": 10, "slot_start": "09:00", "status": "booked"}]
    av = sc.available_slots(doc, booked)
    assert len(av) == 11 and all(s["start"] != "09:00" for s in av)

    # completed visits must free leftover duty time for new bookings
    done_early = [
        {"id": 1, "slot_start": "09:00", "slot_end": "09:20", "status": "done",
         "service_minutes": 8},
    ]
    av2 = sc.available_slots(doc, done_early)
    assert any(s["start_min"] >= sc.to_min("09:08") for s in av2)
    assert len(av2) >= 11
    merged = sc.merge_reclaimed_slots(
        av2, [{"start": "12:40", "start_min": sc.to_min("12:40"),
               "end": "12:55", "end_min": sc.to_min("12:55"),
               "origin": "reclaimed"}],
        now_min=None)
    assert any(s["start"] == "12:40" for s in merged)
    # leftover that already started but has not ended stays bookable
    kept = sc.merge_reclaimed_slots(
        [], [{"start": "18:37", "start_min": sc.to_min("18:37"),
              "end": "18:49", "end_min": sc.to_min("18:49"),
              "origin": "reclaimed"}],
        now_min=sc.to_min("18:41"))
    assert len(kept) == 1 and kept[0]["start"] == "18:37"


def test_scheduling_dynamic_reallocation():
    import scheduling as sc
    # (a) early finish -> frees slots, shrinks the suggested buffer
    doc = {"id": 1, "name": "D", "department": "X", "shift_start": "09:00",
           "shift_end": "11:00", "slot_minutes": 20, "buffer_minutes": 0}
    early = [
        {"id": 1, "patient_name": "a", "priority_level": 0, "status": "done",
         "slot_start": "09:00", "service_minutes": 5, "noshow_prob": 0.1},
        {"id": 2, "patient_name": "b", "priority_level": 0, "status": "done",
         "slot_start": "09:20", "service_minutes": 5, "noshow_prob": 0.1},
    ]
    ls = sc.live_schedule(doc, early, now_min=sc.to_min("09:30"))
    assert ls["summary"]["finishing_early"] and ls["summary"]["new_open_slots"] > 0
    assert ls["summary"]["suggested_buffer_min"] <= 0.1     # overran negative

    # (b) overrun -> projected finish past shift end, larger buffer suggested
    doc2 = {"id": 2, "name": "D2", "department": "X", "shift_start": "09:00",
            "shift_end": "10:00", "slot_minutes": 20, "buffer_minutes": 0}
    late = [
        {"id": 1, "patient_name": "a", "priority_level": 0, "status": "done",
         "slot_start": "09:00", "service_minutes": 40, "noshow_prob": 0.1},
        {"id": 2, "patient_name": "b", "priority_level": 0, "status": "done",
         "slot_start": "09:20", "service_minutes": 40, "noshow_prob": 0.1},
        {"id": 3, "patient_name": "c", "priority_level": 0, "status": "booked",
         "slot_start": "09:40", "noshow_prob": 0.1},
    ]
    ls2 = sc.live_schedule(doc2, late, now_min=sc.to_min("09:50"))
    assert ls2["summary"]["slack_min"] < 0
    assert ls2["summary"]["suggested_buffer_min"] >= 15

    # (c) no-show reclaim -> safe overbooking headroom
    doc3 = {"id": 3, "shift_start": "09:00", "shift_end": "13:00",
            "slot_minutes": 20, "buffer_minutes": 0}
    pend = [{"id": i, "patient_name": str(i), "priority_level": 0,
             "status": "booked",
             "slot_start": f"09:{i * 20:02d}" if i * 20 < 60 else "10:00",
             "noshow_prob": 0.5} for i in range(4)]
    ls3 = sc.live_schedule(doc3, pend, now_min=sc.to_min("08:59"))
    assert ls3["summary"]["safe_overbook"] == 1   # expected 2.0, cap floor(0.3*4)=1

    # (d) leftover duty time after all patients are done is packed using
    # predicted remaining service + department wait-aware buffer (not fixed width)
    leftover_doc = {"id": 4, "name": "D4", "department": "X",
                    "shift_start": "09:00", "shift_end": "11:00",
                    "slot_minutes": 30, "buffer_minutes": 0}
    done_all = [
        {"id": 1, "patient_name": "a", "priority_level": 0, "status": "done",
         "slot_start": "09:00", "service_minutes": 10, "noshow_prob": 0.1},
    ]
    ls4 = sc.live_schedule(
        leftover_doc, done_all, now_min=sc.to_min("09:15"),
        leftover_duration_fn=lambda _t: 12.0,
        leftover_buffer_fn=lambda _t: 3.0,
        predicted_wait=20.0,
    )
    assert ls4["summary"]["new_open_slots"] > 0
    extra = ls4["extra_open_slots"]
    assert extra[0]["service_minutes"] == 12.0
    assert extra[0]["start"] == "09:15"   # leftover starts at now, not before the clock
    # 12+3 step fits the remaining window until 11:00
    assert extra[0]["end_min"] - extra[0]["start_min"] == 12.0

    # late check-in: cannot start before the patient is present
    late_ci = [{
        "id": 1, "patient_name": "late", "priority_level": 0, "status": "checked_in",
        "slot_start": "09:00", "check_in_time": "2026-04-01T09:12:00",
        "noshow_prob": 0.1,
    }]
    ls_late = sc.live_schedule(leftover_doc, late_ci, now_min=sc.to_min("09:15"))
    assert ls_late["timeline"][0]["proj_start"] == "09:12"
    assert ls_late["timeline"][0]["arrival_delay_min"] == 12.0
    # predicted late arrival (not yet checked in) slides the start
    pred_late = [{
        "id": 2, "patient_name": "pred", "priority_level": 0, "status": "booked",
        "slot_start": "09:00", "predicted_arrival_delay": 8.0, "noshow_prob": 0.1,
    }]
    ls_pred = sc.live_schedule(leftover_doc, pred_late, now_min=sc.to_min("08:50"))
    assert ls_pred["timeline"][0]["proj_start"] == "09:08"


def test_store_doctors_and_booking_slots():
    import store
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        did = store.add_doctor(db, {
            "name": "Dr X", "department": "Cardiology", "shift_start": "09:00",
            "shift_end": "12:00", "slot_minutes": 15, "buffer_minutes": 5, "active": 1})
        docs = store.list_doctors(db)
        assert len(docs) == 1 and docs[0]["name"] == "Dr X"
        store.update_doctor(db, did, slot_minutes=20, active=0)
        assert store.get_doctor(db, did)["slot_minutes"] == 20
        assert store.list_doctors(db, active_only=True) == []
        # seeding is a no-op once the roster is populated
        assert store.seed_doctors(db, [{"name": "Dr Y", "department": "General"}]) == 0
        # a booking pinned to a doctor + concrete slot
        bid = store.add_booking(db, {
            "patient_name": "P", "department": "Cardiology", "doctor_id": did,
            "appointment_date": "2026-08-27", "time_slot": "Morning",
            "slot_start": "09:00", "slot_end": "09:15", "day_of_week": "Thursday",
            "priority_level": 0, "noshow_prob": 0.2})
        rows = store.list_bookings(db, doctor_id=did, date="2026-08-27")
        assert len(rows) == 1 and rows[0]["slot_start"] == "09:00"
        store.start_service(db, bid)
        store.complete_service(db, bid, service_minutes=12.0)
        assert store.get_doctor(db, did) is not None
        store.delete_doctor(db, did)
        assert store.get_doctor(db, did) is None


def test_auth_role_and_doctor_roster_link():
    import store
    import service
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        orig = service.db_path
        service.db_path = lambda: Path(db)  # type: ignore[method-assign]
        try:
            store.init_db(db)
            uid, err = service.register_user({
                "username": "docone", "password": "secret", "role": "doctor",
                "full_name": "Dr One", "department": "ENT",
                "shift_start": "09:00", "shift_end": "12:00",
            })
            assert err is None and uid
            docs = store.list_doctors(db)
            assert len(docs) == 1 and docs[0]["name"] == "Dr One"
            user, err = service.authenticate_user("docone", "secret", expected_role="doctor")
            assert err is None and user["role"] == "doctor" and user["doctor_id"] == docs[0]["id"]
            user, err = service.authenticate_user("docone", "secret", expected_role="patient")
            assert user is None and "patient" in (err or "")
        finally:
            service.db_path = orig


def test_pages_compile():
    """Syntax-check every Streamlit file (streamlit itself need not import)."""
    import py_compile
    app_dir = ROOT / "app"
    targets = [app_dir / "streamlit_app.py", app_dir / "service.py",
               app_dir / "store.py", app_dir / "scheduling.py"]
    targets += sorted((app_dir / "pages").glob("*.py"))
    checked = 0
    for f in targets:
        if f.exists():
            py_compile.compile(str(f), doraise=True)
            checked += 1
    assert checked >= 4


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
