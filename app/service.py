"""Service layer bridging the Streamlit UI to the QMS package.

Responsibilities
----------------
* locate the project root and put ``src/`` on the path (so ``import qms`` works
  from a plain checkout, no install required);
* load config, trained model bundles, and evaluation artifacts (all optional --
  the UI degrades gracefully if training/eval hasn't been run yet);
* build *complete, realistic* feature rows for on-the-fly prediction by starting
  from dataset medians/modes and overriding with the user's form inputs;
* expose no-show and waiting-time predictions and a GenAI explainer.

Importable outside Streamlit (used by the smoke test), so caching falls back to
plain memoisation when Streamlit isn't present.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# --- locate root and wire up imports --------------------------------------
_HERE = Path(__file__).resolve()
ROOT = next(p for p in _HERE.parents if (p / "config" / "config.yaml").exists())
for _p in (ROOT / "src", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import store                                            # noqa: E402
import scheduling                                       # noqa: E402
from qms.config import load_config, abspath           # noqa: E402
from qms.data import preprocess                        # noqa: E402
from qms.models.estimators import ModelBundle          # noqa: E402
from qms.genai.explainer import get_explainer          # noqa: E402


@lru_cache(maxsize=1)
def get_cfg() -> dict:
    return load_config()


def db_path() -> Path:
    return abspath(get_cfg()["app"]["db_path"])


# --- dataset-derived defaults & form options ------------------------------
@lru_cache(maxsize=1)
def _clean_df() -> Optional[pd.DataFrame]:
    path = abspath(get_cfg()["data"]["clean_csv"])
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


@lru_cache(maxsize=1)
def feature_defaults() -> Dict[str, object]:
    """Median (numeric) / mode (categorical) for every model input column."""
    df = _clean_df()
    cols = set(preprocess.BOOKING_NUMERIC + preprocess.ARRIVAL_NUMERIC
               + preprocess.CHECKIN_NUMERIC + preprocess.SERVICE_NUMERIC
               + preprocess.BOOKING_CATEGORICAL + preprocess.CHECKIN_CATEGORICAL)
    defaults: Dict[str, object] = {}
    if df is None:
        return {c: (0.0 if c not in ("department", "day_of_week", "time_slot")
                    else "unknown") for c in cols}
    for c in cols:
        if c not in df.columns:
            defaults[c] = 0.0
            continue
        s = df[c]
        if c in preprocess.BOOKING_CATEGORICAL or c in preprocess.CHECKIN_CATEGORICAL:
            m = s.dropna().mode()
            defaults[c] = str(m.iloc[0]) if len(m) else "unknown"
        else:
            defaults[c] = float(pd.to_numeric(s, errors="coerce").median())
    return defaults


@lru_cache(maxsize=1)
def options() -> Dict[str, List[str]]:
    """Choices for the booking form (from the data profile / clean data)."""
    prof_path = abspath(get_cfg()["data"]["profile_json"])
    depts: List[str] = []
    if prof_path.exists():
        try:
            depts = json.loads(prof_path.read_text()).get("departments", []) or []
        except Exception:  # noqa: BLE001
            depts = []
    df = _clean_df()
    if not depts and df is not None and "department" in df:
        depts = sorted(df["department"].dropna().unique().tolist())
    slots = (sorted(df["time_slot"].dropna().unique().tolist())
             if df is not None and "time_slot" in df else
             ["Morning", "Afternoon", "Evening"])
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
    return {"departments": depts or ["General"], "time_slots": slots, "days": days}


# trained bundles must be re-read after scripts/02_train_models.py
_BUNDLES: Dict[str, object] = {}


def load_bundle(task: str) -> Optional[ModelBundle]:
    path = abspath(get_cfg()["models"]["dir"]) / f"{task}.pkl"
    mtime = path.stat().st_mtime if path.exists() else None
    cached = _BUNDLES.get(task)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    bundle = None
    if path.exists():
        try:
            bundle = ModelBundle.load(path)
        except Exception:  # noqa: BLE001 — pickle/sklearn version mismatch
            bundle = None
    _BUNDLES[task] = (mtime, bundle)
    return bundle


def has_models() -> bool:
    return all(load_bundle(t) is not None
               for t in ("no_show", "arrival_delay", "waiting_time"))


def has_service_model() -> bool:
    return load_bundle("service_time") is not None


def _clip_minutes(value: float, lo: float = 5.0, hi: float = 90.0) -> float:
    cfg = get_cfg().get("roster", {})
    lo = float(cfg.get("min_service_minutes", lo))
    hi = float(cfg.get("max_service_minutes", hi))
    return float(max(lo, min(hi, value)))


def _coarse_slot(hour: int) -> str:
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    return "Evening"


@lru_cache(maxsize=1)
def department_stats() -> Dict[str, Dict[str, float]]:
    """Median service and mean wait per department from historical data."""
    df = _clean_df()
    out: Dict[str, Dict[str, float]] = {}
    if df is None:
        return out

    def _agg(frame) -> Dict[str, float]:
        svc = pd.to_numeric(frame.get("service_time"), errors="coerce")
        wait = pd.to_numeric(frame.get("waiting_time"), errors="coerce")
        delay = pd.to_numeric(frame.get("arrival_delay"), errors="coerce")
        if "no_show" in frame.columns:
            shown = pd.to_numeric(frame["no_show"], errors="coerce").fillna(0).eq(0)
            delay = delay.where(shown)
        return {
            "median_service": float(svc.median()) if svc.notna().any() else 15.0,
            "mean_wait": float(wait.mean()) if wait.notna().any() else 0.0,
            "p90_wait": float(wait.quantile(0.9)) if wait.notna().any() else 0.0,
            "mean_arrival_delay": float(delay.mean()) if delay.notna().any() else 0.0,
        }

    if "department" in df.columns:
        for dept, g in df.groupby(df["department"].astype(str)):
            out[str(dept)] = _agg(g)
    out["_all"] = _agg(df)
    return out


def dept_duration_profile(department: Optional[str]) -> Dict[str, float]:
    stats = department_stats()
    if department and department in stats:
        return stats[department]
    return stats.get("_all", {"median_service": 15.0, "mean_wait": 0.0,
                              "p90_wait": 0.0, "mean_arrival_delay": 0.0})


def predict_service(inputs: Dict) -> float:
    """Predicted consult minutes from the service-time model only.

    Returns NaN when the model is missing so callers can refuse fixed-width packing.
    """
    bundle = load_bundle("service_time")
    if bundle is None:
        return float("nan")
    X = _row_from(inputs, preprocess.SERVICE_NUMERIC, preprocess.SERVICE_CATEGORICAL)
    pred = float(bundle.predict(X)[0])
    if pred != pred:
        return float("nan")
    return _clip_minutes(pred)


def _buffer_cfg() -> Dict:
    return get_cfg().get("buffer", {}) or {}


def _queue_pressure(bookings: List[Dict]) -> Dict[str, float]:
    """Live crowd signals used by the dynamic buffer."""
    waiting = sum(1 for b in bookings if b.get("status") == "checked_in")
    pending = sum(1 for b in bookings if b.get("status") in ("booked", "checked_in"))
    in_service = sum(1 for b in bookings if b.get("status") == "in_service")
    probs = [
        float(b["noshow_prob"]) for b in bookings
        if b.get("status") in ("booked", "checked_in")
        and b.get("noshow_prob") not in (None, "")
    ]
    return {
        "waiting_count": float(waiting),
        "pending_count": float(pending),
        "in_service_count": float(in_service),
        "noshow_probs": probs,
    }


def _dynamic_buffer_minutes(department: Optional[str], *,
                            noshow_probs: Optional[List[float]] = None,
                            mean_overrun: float = 0.0,
                            waiting_count: int = 0,
                            pending_count: int = 0,
                            slack_min: float = 0.0,
                            predicted_wait: Optional[float] = None) -> float:
    """Crowd-aware inter-slot buffer (minutes). Not a static clinic gap."""
    profile = dept_duration_profile(department)
    cfg = _buffer_cfg()
    from qms.optimization.buffer import DynamicBuffer
    engine = DynamicBuffer(
        base_minutes=float(cfg.get("base_minutes", 10)),
        max_overbook_fraction=float(cfg.get("max_overbook_fraction", 0.30)),
    )
    wait = (predicted_wait if predicted_wait is not None
            else profile.get("mean_wait", 0.0))
    return engine.slot_buffer(
        predicted_wait=float(wait or 0.0),
        mean_overrun=mean_overrun,
        noshow_probs=noshow_probs or [],
        waiting_count=int(waiting_count or 0),
        pending_count=int(pending_count or 0),
        slack_min=float(slack_min or 0.0),
        crowd_compress_per_waiter=float(cfg.get("crowd_compress_per_waiter", 0.75)),
        crowd_compress_cap=float(cfg.get("crowd_compress_cap", 6.0)),
        behind_compress_per_min=float(cfg.get("behind_compress_per_min", 0.05)),
        min_minutes=float(cfg.get("min_minutes", 1.0)),
        max_minutes=float(cfg.get("max_minutes", 20.0)),
    )


def doctor_with_duty(doctor: Dict, date_iso: str) -> Dict:
    """Overlay published duty-day shift hours onto the doctor roster record."""
    out = dict(doctor)
    duty = store.get_doctor_duty(db_path(), int(doctor["id"]), date_iso)
    if duty and duty.get("is_available"):
        if duty.get("shift_start"):
            out["shift_start"] = duty["shift_start"]
        if duty.get("shift_end"):
            out["shift_end"] = duty["shift_end"]
        out["duty_published"] = True
    else:
        out["duty_published"] = bool(duty and duty.get("is_available"))
    return out


def _slot_fns(doctor: Dict, date_iso: str):
    """Closures that size the next slot from the service-time model + crowd buffer.

    Doctor ``slot_minutes`` is never used as a clinic-wide width. Buffer shrinks
    under live queue pressure and widens when consults overrun.
    """
    from datetime import date as _date
    try:
        day_name = _date.fromisoformat(date_iso).strftime("%A")
        is_weekend = 1.0 if day_name in ("Saturday", "Sunday") else 0.0
    except ValueError:
        day_name, is_weekend = "Monday", 0.0
    dept = doctor.get("department")
    bookings = store.list_bookings(db_path(), doctor_id=doctor["id"], date=date_iso)
    pressure = _queue_pressure(bookings)
    profile = dept_duration_profile(dept)
    # Live waiting-time model when available, else department history.
    live_wait = profile.get("mean_wait", 0.0)
    if pressure["waiting_count"] > 0 and load_bundle("waiting_time") is not None:
        pred_w = predict_wait({
            "department": dept,
            "day_of_week": day_name,
            "time_slot": _coarse_slot(int(_now_min() // 60) % 24),
            "hour_of_day": int(_now_min() // 60) % 24,
            "is_weekend": is_weekend,
            "priority_level": 0,
            "active_doctors_count": 1,
            "queue_length": pressure["waiting_count"],
        })
        if pred_w == pred_w:
            live_wait = float(pred_w)

    buf = _dynamic_buffer_minutes(
        dept,
        noshow_probs=pressure["noshow_probs"],
        waiting_count=int(pressure["waiting_count"]),
        pending_count=int(pressure["pending_count"]),
        predicted_wait=live_wait,
    )

    def duration_fn(start_min: float, booking: Optional[Dict] = None) -> float:
        # If realised service duration is recorded (for completed visits), use it
        if booking and booking.get("service_minutes") not in (None, ""):
            try:
                actual = float(booking["service_minutes"])
                if actual > 0:
                    return actual
            except (ValueError, TypeError):
                pass

        hour = int(start_min // 60) % 24
        p_level = int(booking.get("priority_level", 0)) if booking else 0
        b_dept = (booking.get("department") if booking else None) or dept
        b_slot = (booking.get("time_slot") if booking else None) or _coarse_slot(hour)

        pred = predict_service({
            "department": b_dept,
            "day_of_week": day_name,
            "time_slot": b_slot,
            "hour_of_day": hour,
            "is_weekend": is_weekend,
            "is_peak_hour": 1.0 if 9 <= hour <= 12 else 0.0,
            "priority_level": p_level,
            "active_doctors_count": 1,
            "slot_arrival_count": max(1, int(pressure["pending_count"]) or 1),
        })
        return pred if pred == pred else 0.0

    def buffer_fn(_start_min: float) -> float:
        return buf

    return duration_fn, buffer_fn, buf


def preview_shift_plan(doctor: Dict, date_iso: Optional[str] = None) -> List[Dict]:
    """Slots generated from published duty hours using the ML duration model."""
    from datetime import date as _date
    date_iso = date_iso or _date.today().isoformat()
    doctor = doctor_with_duty(doctor, date_iso)
    duration_fn, buffer_fn, _ = _slot_fns(doctor, date_iso)
    return scheduling.generate_shift_slots(
        doctor.get("shift_start"), doctor.get("shift_end"),
        duration_fn=duration_fn, buffer_fn=buffer_fn, origin="planned")


def backend_label() -> str:
    b = load_bundle("no_show")
    return b.backend if b else "not trained"


def _row_from(inputs: Dict, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    """One-row raw feature frame: defaults overridden by provided inputs."""
    d = dict(feature_defaults())
    for k, v in inputs.items():
        if v is not None:
            d[k] = v
    raw = pd.DataFrame([d])
    raw = preprocess.add_derived_features(raw)
    row = {c: raw.iloc[0][c] if c in raw.columns else d.get(c, 0.0)
           for c in numeric + categorical}
    return pd.DataFrame([row])


# --- predictions ----------------------------------------------------------
def predict_no_show(inputs: Dict) -> Tuple[float, List[str]]:
    """Return (probability, human-readable driver list) for a booking."""
    bundle = load_bundle("no_show")
    X = _row_from(inputs, preprocess.BOOKING_NUMERIC, preprocess.BOOKING_CATEGORICAL)
    prob = float(bundle.predict_proba(X)[0]) if bundle else float("nan")

    drivers: List[str] = []
    pnc = inputs.get("previous_no_show_count", 0) or 0
    if pnc >= 2:
        drivers.append(f"history of {int(pnc)} prior no-shows")
    elif pnc == 1:
        drivers.append("one prior no-show on record")
    lt = inputs.get("lead_time_hours")
    if lt is not None and lt >= 168:
        drivers.append("long lead time (booked well in advance)")
    if inputs.get("time_slot"):
        drivers.append(f"{inputs['time_slot'].lower()} slot")
    if bundle and bundle.feature_importances and len(drivers) < 3:
        for feat in list(bundle.feature_importances)[:3]:
            nice = feat.split("_")[0]
            if nice not in " ".join(drivers):
                drivers.append(f"model signal: {feat}")
            if len(drivers) >= 3:
                break
    return prob, drivers[:3]


def predict_wait(inputs: Dict) -> float:
    bundle = load_bundle("waiting_time")
    X = _row_from(inputs, preprocess.CHECKIN_NUMERIC, preprocess.CHECKIN_CATEGORICAL)
    return float(bundle.predict(X)[0]) if bundle else float("nan")


def predict_arrival_delay(inputs: Dict) -> float:
    """Predicted minutes early (negative) / late (positive) at the booked slot."""
    filled = dict(inputs)
    profile = dept_duration_profile(filled.get("department"))
    hist = float(profile.get("mean_arrival_delay", 0.0) or 0.0)
    filled.setdefault("dept_mean_delay", hist)
    filled.setdefault("hour_mean_delay", hist)
    bundle = load_bundle("arrival_delay")
    X = _row_from(filled, preprocess.ARRIVAL_NUMERIC, preprocess.BOOKING_CATEGORICAL)
    return float(bundle.predict(X)[0]) if bundle else float("nan")


# --- evaluation artifacts (for dashboards) --------------------------------
def _read_json(rel_name: str) -> Optional[dict]:
    path = abspath(get_cfg()["reports"]["metrics_dir"]) / rel_name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def policy_comparison() -> Optional[dict]:
    return _read_json("policy_comparison.json")


def buffer_report() -> Optional[dict]:
    return _read_json("buffer_analysis.json")


def model_metrics() -> Optional[dict]:
    return _read_json("model_metrics.json")


def conformal_metrics() -> Optional[dict]:
    return _read_json("conformal_metrics.json")


def pareto_report() -> Optional[dict]:
    return _read_json("pareto_analysis.json")


def chance_constrained_report() -> Optional[dict]:
    return _read_json("chance_constrained_buffer.json")


def fairness_report() -> Optional[dict]:
    return _read_json("fairness_audit.json")


def counterfactual_report() -> Optional[dict]:
    return _read_json("counterfactual_scenarios.json")


def figure_path(name: str) -> Optional[Path]:
    p = abspath(get_cfg()["reports"]["figures_dir"]) / name
    return p if p.exists() else None


def explainer():
    """Fresh explainer so a session-pasted Gemini key is picked up."""
    try:
        import streamlit as st
        key = st.session_state.get("gemini_api_key")
        if key:
            import os
            os.environ["GEMINI_API_KEY"] = key
    except Exception:  # noqa: BLE001
        pass
    return get_explainer(get_cfg())


# --- doctors / roster -----------------------------------------------------
def list_doctors(department: Optional[str] = None,
                 active_only: bool = True) -> List[Dict]:
    return store.list_doctors(db_path(), department=department,
                              active_only=active_only)


def list_all_hospital_doctors() -> List[Dict]:
    """Return all hospital doctors (both active and inactive) with their login credentials."""
    seed_default_system()
    return store.list_all_hospital_doctors(db_path())


def add_doctor_with_login(doctor_rec: Dict, username: str, password: str) -> Tuple[int, Optional[str]]:
    """Create a new doctor and their linked user account."""
    seed_default_system()
    return store.add_doctor_with_login(db_path(), doctor_rec, username, password)


def set_doctor_status(doctor_id: int, active: int | bool) -> None:
    """Toggle a doctor's active status."""
    seed_default_system()
    store.set_doctor_status(db_path(), doctor_id, active)


def set_doctor_availability(doctor_id: int, available_date: str, is_available: bool | int) -> None:
    """Compatibility wrapper — duty days are published by doctors, not admins."""
    seed_default_system()
    store.set_doctor_availability(db_path(), int(doctor_id), available_date, is_available)


def publish_doctor_duty(
    user: Optional[Dict],
    doctor_id: int,
    available_date: str,
    *,
    is_available: bool | int = True,
    shift_start: Optional[str] = None,
    shift_end: Optional[str] = None,
) -> Dict:
    """Doctor publishes on-duty hours; ML slots open inside that window."""
    require_role(user, ("doctor", "admin"))
    seed_default_system()
    if user and str(user.get("role", "")).lower() == "doctor":
        linked = int(user.get("doctor_id") or 0)
        if linked and linked != int(doctor_id):
            raise PermissionError("Doctors may only publish their own duty hours.")
    return store.publish_doctor_duty(
        db_path(), int(doctor_id), available_date,
        is_available=is_available,
        shift_start=shift_start,
        shift_end=shift_end,
        published_by=f"{(user or {}).get('role', 'doctor')}:{(user or {}).get('username', 'user')}",
    )


def is_doctor_available(doctor_id: int, available_date: str) -> bool:
    """Active doctor who has published on-duty hours for the date."""
    seed_default_system()
    return store.is_doctor_available(db_path(), int(doctor_id), available_date)


def is_doctor_available_on_date(doctor_id: int, date_iso: str) -> Tuple[bool, str]:
    """Check if a doctor is active and has published duty hours for the date.

    Returns (status: bool, reason: str).
    """
    seed_default_system()
    db = db_path()
    doc = store.get_doctor(db, int(doctor_id))
    if not doc:
        return False, "Doctor not found."

    if not doc.get("active"):
        return False, f"Doctor {doc.get('name', 'Doctor')} is currently inactive and unavailable for appointments."

    is_avail = store.is_doctor_available(db, int(doctor_id), date_iso)
    if not is_avail:
        return False, (
            f"{doc.get('name', 'Doctor')} has not published duty hours for this date. "
            f"Slots open only after the doctor sets their shift for the day."
        )

    return True, f"{doc.get('name', 'Doctor')} is on duty on {date_iso}."


def list_doctor_availability(doctor_id: Optional[int] = None, available_date: Optional[str] = None) -> List[Dict]:
    """Retrieve doctor-published duty records (read-only for admin dashboards)."""
    seed_default_system()
    return store.list_doctor_availability(db_path(), doctor_id=doctor_id, available_date=available_date)


def list_doctor_published_dates(doctor_id: Optional[int] = None, department: Optional[str] = None, from_date: Optional[str] = None) -> List[Dict]:
    """Return dates where doctor(s) have published on-duty shifts."""
    seed_default_system()
    return store.list_doctor_published_dates(db_path(), doctor_id=doctor_id, department=department, from_date=from_date)


def list_available_doctors_for_date(department: Optional[str] = None, date_str: Optional[str] = None) -> List[Dict]:
    """Return doctors who are BOTH active and available on the specified date."""
    seed_default_system()
    return store.list_available_doctors(db_path(), department=department, available_date=date_str)


def get_doctor(doctor_id: int) -> Optional[Dict]:
    return store.get_doctor(db_path(), int(doctor_id))


def doctor_departments() -> List[str]:
    """Departments that currently have at least one active doctor."""
    return sorted({d["department"] for d in list_doctors(active_only=True)})


def specialty_departments() -> List[str]:
    """Catalog of specialties (dataset), combined with the live roster."""
    catalog = set(d for d in (options().get("departments") or []) if d)
    live_depts = set(doctor_departments() or [])
    combined = sorted(catalog | live_depts)
    return combined or ["General"]


def _dept_service_minutes(dept: str) -> float:
    """Median realised service time for a department (fallback 15 min).

    Used only to seed a *sensible default* slot width for new doctors — the
    admin can override it per doctor.
    """
    df = _clean_df()
    if df is None or "service_time" not in df or "department" not in df:
        return 15.0
    sub = pd.to_numeric(df.loc[df["department"] == dept, "service_time"],
                        errors="coerce").median()
    if sub != sub:
        sub = pd.to_numeric(df["service_time"], errors="coerce").median()
    return float(round(sub)) if sub == sub else 15.0


def default_roster() -> List[Dict]:
    """No fake roster. Doctors are created when a doctor account is registered."""
    return []


def seed_default_doctors() -> int:
    return 0


# --- slots: booking-time availability + live re-allocation ----------------
def _now_min() -> float:
    from datetime import datetime
    n = datetime.now()
    return n.hour * 60 + n.minute + n.second / 60.0


def available_slots(doctor: Dict, date_iso: str) -> List[Dict]:
    """Open slots packed from the service-time model inside published duty hours.

    Early finishes free leftover duty time; ``live_schedule`` reclaim merges
    those extra ML-sized slots into the bookable set automatically.
    """
    from datetime import date as _date
    if not has_service_model():
        return []
    if not doctor.get("active"):
        return []
    if not is_doctor_available(doctor["id"], date_iso):
        return []
    doctor = doctor_with_duty(doctor, date_iso)
    bookings = store.list_bookings(db_path(), doctor_id=doctor["id"], date=date_iso)
    is_today = date_iso == _date.today().isoformat()
    now = _now_min() if is_today else None
    duration_fn, buffer_fn, _ = _slot_fns(doctor, date_iso)
    profile = dept_duration_profile(doctor.get("department"))

    packed = scheduling.available_slots(
        doctor, bookings, now_min=now if is_today else None,
        is_today=is_today, duration_fn=duration_fn, buffer_fn=buffer_fn)

    ls = scheduling.live_schedule(
        doctor, bookings, now_min=now if is_today else None,
        max_overbook_fraction=_buffer_cfg().get("max_overbook_fraction", 0.30),
        remaining_service_fn=duration_fn,
        leftover_duration_fn=duration_fn,
        leftover_buffer_fn=buffer_fn,
        predicted_wait=profile.get("mean_wait", 0.0))

    extra = ls.get("extra_open_slots") or []
    candidate_slots = scheduling.merge_reclaimed_slots(packed, extra, now_min=now)

    valid_slots: List[Dict] = []
    for s in sorted(candidate_slots, key=lambda x: x.get("start_min", 0)):
        s_start = s.get("slot_start") or s.get("start")
        s_end = s.get("slot_end") or s.get("end")
        sm = scheduling.to_min(s_start)
        em = scheduling.to_min(s_end)
        if sm is None or em is None or em <= sm:
            continue
        if store.check_booking_conflict(db_path(), doctor["id"], date_iso, s_start, s_end):
            continue
        if any(max(sm, o.get("start_min", 0)) < min(em, o.get("end_min", 0)) - 1e-6 for o in valid_slots):
            continue
        # Annotate the live dynamic buffer used for this open slot
        s.setdefault("buffer_minutes", buffer_fn(sm))
        valid_slots.append(s)
    return valid_slots


def live_schedule(doctor: Dict | int, date_iso: str) -> Dict:
    """The doctor's live, dynamically re-allocated schedule for a date."""
    from datetime import date as _date
    if isinstance(doctor, (int, str)):
        doctor = get_doctor(int(doctor))
    if not doctor or not doctor.get("active") or not is_doctor_available(doctor["id"], date_iso):
        return {
            "doctor": doctor or {},
            "now": "",
            "now_12h": "",
            "timeline": [],
            "extra_open_slots": [],
            "summary": {
                "available": False,
                "planned_slot_count": 0,
                "booked_count": 0,
                "checked_in_count": 0,
                "in_service_count": 0,
                "done_count": 0,
                "total_scheduled": 0,
                "projected_finish": "Unavailable",
                "projected_finish_12h": "Unavailable",
                "shift_end": "",
                "shift_end_12h": "",
                "slack_min": 0.0,
                "finishing_early": False,
                "new_open_slots": 0,
                "expected_no_shows": 0.0,
                "safe_overbook": 0,
                "mean_overrun_min": 0.0,
                "time_saved_min": 0.0,
                "emergency_count": 0,
                "suggested_buffer_min": 0.0,
                "predicted_service_min": 0.0,
                "dept_avg_wait_min": 0.0,
                "buffer_mode": "dynamic_crowd_aware",
            },
        }
    doctor = doctor_with_duty(doctor, date_iso)
    bookings = store.list_bookings(db_path(), doctor_id=doctor["id"], date=date_iso)
    is_today = date_iso == _date.today().isoformat()
    duration_fn, buffer_fn, live_buf = _slot_fns(doctor, date_iso)
    profile = dept_duration_profile(doctor.get("department"))
    day_name = ""
    try:
        day_name = _date.fromisoformat(date_iso).strftime("%A")
    except ValueError:
        day_name = ""
    bookings = _with_arrival_preds(bookings, day_name)
    result = scheduling.live_schedule(
        doctor, bookings, now_min=_now_min() if is_today else None,
        max_overbook_fraction=_buffer_cfg().get("max_overbook_fraction", 0.30),
        remaining_service_fn=duration_fn,
        leftover_duration_fn=duration_fn,
        leftover_buffer_fn=buffer_fn,
        predicted_wait=profile.get("mean_wait", 0.0),
        is_today=is_today,
        date_iso=date_iso)
    # Recompute crowd-aware suggested buffer with live slack after roll-forward
    pressure = _queue_pressure(bookings)
    summary = result.setdefault("summary", {})
    crowd_buf = _dynamic_buffer_minutes(
        doctor.get("department"),
        noshow_probs=pressure["noshow_probs"],
        mean_overrun=float(summary.get("mean_overrun_min") or 0.0),
        waiting_count=int(pressure["waiting_count"]),
        pending_count=int(pressure["pending_count"]),
        slack_min=float(summary.get("slack_min") or 0.0),
        predicted_wait=profile.get("mean_wait", 0.0),
    )
    summary["suggested_buffer_min"] = crowd_buf
    summary["live_buffer_min"] = live_buf
    summary["waiting_count"] = int(pressure["waiting_count"])
    summary["buffer_mode"] = "dynamic_crowd_aware"
    summary["available"] = True
    # If early finish left duty time, re-pack leftover with the refreshed buffer
    if summary.get("finishing_early") and summary.get("new_open_slots", 0) == 0:
        leftover_from = scheduling.to_min(summary.get("projected_finish"))
        if leftover_from is not None:
            extra = scheduling.generate_shift_slots(
                doctor.get("shift_start"), doctor.get("shift_end"),
                duration_fn=duration_fn,
                buffer_fn=lambda _t: crowd_buf,
                from_min=leftover_from, occupied=[], origin="reclaimed")
            result["extra_open_slots"] = extra
            summary["new_open_slots"] = len(extra)
    return result


def _with_arrival_preds(bookings: List[Dict], day_name: str) -> List[Dict]:
    """Fill predicted arrival delay for booked patients that lack a stored score."""
    out = []
    for b in bookings:
        row = dict(b)
        if (row.get("status") == "booked"
                and row.get("predicted_arrival_delay") in (None, "")
                and load_bundle("arrival_delay") is not None):
            hr = 9
            start = scheduling.to_min(row.get("slot_start"))
            if start is not None:
                hr = int(start // 60) % 24
            pred = predict_arrival_delay({
                "department": row.get("department"),
                "time_slot": row.get("time_slot") or _coarse_slot(hr),
                "day_of_week": row.get("day_of_week") or day_name,
                "priority_level": row.get("priority_level") or 0,
                "lead_time_hours": row.get("lead_time_hours") or 0,
                "previous_no_show_count": row.get("previous_no_show_count") or 0,
                "hour_of_day": hr,
            })
            if pred == pred:
                row["predicted_arrival_delay"] = pred
        out.append(row)
    return out


# =====================================================================
# Dual-Role Authentication & Patient Session Helpers
# =====================================================================
def seed_default_system() -> None:
    """Open the SQLite store."""
    store.init_db(db_path())


def require_role(user: Optional[Dict], allowed_roles: Sequence[str]) -> None:
    """Enforce server-side role check; raises PermissionError if unauthorized."""
    if not user:
        raise PermissionError("Authentication required.")
    role = str(user.get("role", "")).lower()
    if role not in [r.lower() for r in allowed_roles]:
        raise PermissionError(f"Access denied: role '{role}' is not in authorized roles {list(allowed_roles)}.")


def validate_patient_access(user: Optional[Dict], patient_user_id: int) -> None:
    """Enforce patient data isolation. Admins and attending doctors may view, but patients can only access their own."""
    if not user:
        raise PermissionError("Authentication required.")
    role = str(user.get("role", "")).lower()
    if role in ("admin", "doctor"):
        return
    if role == "patient":
        if int(user.get("id") or 0) != int(patient_user_id):
            raise PermissionError("Security alert: Unauthorized access to another patient's medical records.")
    else:
        raise PermissionError("Access denied.")


def authenticate_user(username: str, password: str,
                      expected_role: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """Return (user, error). ``expected_role`` enforces the portal the user opened."""
    seed_default_system()
    user = store.authenticate_user(db_path(), username, password)
    if not user:
        return None, "Invalid username or password."
    if expected_role and user.get("role") != expected_role:
        return None, (
            f"This account is registered as **{user.get('role')}**. "
            f"Please sign in on the {user.get('role')} portal (not the {expected_role} portal)."
        )
    return user, None


def register_user(user_dict: Dict) -> Tuple[Optional[int], Optional[str]]:
    seed_default_system()
    rec = dict(user_dict)
    role = str(rec.get("role") or "patient").strip().lower()
    if role == "doctor" and not rec.get("doctor_id"):
        did = store.add_doctor(db_path(), {
            "name": rec.get("full_name") or rec.get("username"),
            "department": rec.get("department") or "General",
            "shift_start": rec.get("shift_start") or "09:00",
            "shift_end": rec.get("shift_end") or "13:00",
            "slot_minutes": 0,
            "buffer_minutes": float(get_cfg()["buffer"]["base_minutes"]),
            "active": 1,
        })
        rec["doctor_id"] = did
    return store.create_user(db_path(), rec)


def get_user_by_id(user_id: int) -> Optional[Dict]:
    return store.get_user_by_id(db_path(), user_id)


def list_patient_bookings(patient_user_id: Optional[int] = None,
                          patient_name: Optional[str] = None) -> List[Dict]:
    seed_default_system()
    return store.list_bookings(db_path(), patient_user_id=patient_user_id,
                               patient_name=patient_name)


def get_patient_medical_history(user: Optional[Dict], patient_user_id: int) -> List[Dict]:
    """Retrieve full consultation history with doctor notes; validates patient isolation."""
    seed_default_system()
    validate_patient_access(user, patient_user_id)
    return store.get_patient_medical_history(db_path(), patient_user_id)


def get_booking(booking_id: int) -> Optional[Dict]:
    """Retrieve booking by ID."""
    seed_default_system()
    return store.get_booking(db_path(), int(booking_id))


def get_patient_live_ticket(patient_user_id: Optional[int] = None,
                            patient_name: Optional[str] = None,
                            booking_id: Optional[int] = None) -> Optional[Dict]:
    """Retrieve the active/relevant appointment ticket with live queue position & dynamic ETA for patient.
    Excludes past and completed appointments from the default live queue ticket.
    """
    from datetime import date as _date
    all_b = list_patient_bookings(patient_user_id=patient_user_id, patient_name=patient_name)
    if not all_b and booking_id is None:
        return None

    today_iso = _date.today().isoformat()
    now_m = _now_min()

    target = None

    if booking_id is not None:
        # User explicitly requested a specific booking ticket
        matched = [b for b in all_b if int(b.get("id", 0)) == int(booking_id)]
        if matched:
            target = dict(matched[0])
        else:
            b_rec = store.get_booking(db_path(), int(booking_id))
            if b_rec:
                if patient_user_id and b_rec.get("patient_user_id") and int(b_rec["patient_user_id"]) != int(patient_user_id):
                    return None
                target = dict(b_rec)
        if not target:
            return None
    else:
        # Default Live Queue Ticket:
        # Must exclude completed appointments, past appointments, and cancelled/no_show.
        # Only active upcoming/current appointments (in_service, checked_in today, booked today, or upcoming future)
        active_candidates = [
            b for b in all_b
            if b.get("status") in ("in_service", "checked_in", "booked")
        ]
        if not active_candidates:
            # Issue 2: Completed past appointments must not appear in the active Live Queue
            return None

        def _ticket_rank(b: Dict):
            st = b.get("status", "booked")
            appt_date = b.get("appointment_date", "")
            is_today = (appt_date == today_iso)
            is_past = bool(appt_date and appt_date < today_iso)

            if is_past:
                rank = 10
            elif st == "in_service":
                rank = 0
            elif st == "checked_in" and is_today:
                rank = 1
            elif st == "booked" and is_today:
                rank = 2
            elif st == "checked_in":
                rank = 3
            elif st == "booked":
                rank = 4
            else:
                rank = 5

            try:
                date_num = int(appt_date.replace("-", ""))
            except ValueError:
                date_num = 0

            return (rank, date_num, b.get("slot_start") or "", -int(b.get("id") or 0))

        sorted_b = sorted(active_candidates, key=_ticket_rank)
        target = dict(sorted_b[0])

    status = target.get("status", "booked")
    target_appt_date = target.get("appointment_date", "")
    doctor_id = target.get("doctor_id")

    # Format human-readable date and time
    target["appointment_date_human"] = scheduling.format_date_human(target_appt_date)
    target["appointment_time_12h"] = scheduling.format_clock_12h(target.get("slot_start"))
    target["slot_start_12h"] = target["appointment_time_12h"]

    doc = store.get_doctor(db_path(), doctor_id) if doctor_id else None
    timeline = []
    e_12h = "—"
    if doc:
        target["doctor_name"] = doc.get("name")
        target["doctor_department"] = doc.get("department")
        s_12h = scheduling.format_clock_12h(doc.get("shift_start"))
        e_12h = scheduling.format_clock_12h(doc.get("shift_end"))
        target["doctor_shift"] = f"{s_12h} - {e_12h}"
        ls = live_schedule(doc, target_appt_date)
        timeline = ls.get("timeline", [])
        summary = ls.get("summary", {})
        target["doctor_slack_min"] = summary.get("slack_min", 0.0)
        target["doctor_projected_finish"] = summary.get("projected_finish_12h") or summary.get("projected_finish", "—")
    else:
        target["doctor_slack_min"] = 0.0
        target["doctor_projected_finish"] = "—"

    item = next((it for it in timeline if it.get("id") == target.get("id")), None)
    is_target_past = bool(target_appt_date) and (target_appt_date < today_iso)

    if status == "booked":
        target["queue_position"] = "Available after check-in"
        target["queue_position_display"] = "Available after check-in"
        target["queue_pos_num"] = None
        target["patients_ahead"] = None
        target["patients_ahead_display"] = "--"
        target["live_eta_minutes"] = None
        target["live_eta_display"] = "Available after check-in"
        target["estimated_turn"] = target["appointment_time_12h"]
        target["estimated_turn_display"] = target["appointment_time_12h"]
        target["is_live_queue"] = False
        target["notifications"] = []
        if item:
            target["proj_start"] = item.get("proj_start")
            target["proj_end"] = item.get("proj_end")
            target["delay_min"] = item.get("delay_min", 0.0)

    elif status == "checked_in":
        if is_target_past:
            target["queue_position"] = "Expired"
            target["queue_position_display"] = "Expired"
            target["queue_pos_num"] = None
            target["patients_ahead"] = None
            target["patients_ahead_display"] = "--"
            target["live_eta_minutes"] = None
            target["live_eta_display"] = "Expired"
            target["estimated_turn"] = target["appointment_time_12h"]
            target["estimated_turn_display"] = target["appointment_time_12h"]
            target["is_live_queue"] = False
            target["notifications"] = []
        else:
            # Patient is in active live queue
            if item:
                target["queue_position"] = item.get("queue_position", "#1")
                target["queue_position_display"] = item.get("queue_position", "#1")
                target["queue_pos_num"] = item.get("queue_pos_num", 1)
                target["patients_ahead"] = item.get("patients_ahead", 0)
                target["patients_ahead_display"] = str(item.get("patients_ahead", 0))
                target["live_eta_minutes"] = item.get("est_wait_min", 0.0)
                wait_val = item.get("est_wait_min", 0.0)
                target["live_eta_display"] = f"{float(wait_val):.0f} minutes" if wait_val is not None else "0 minutes"
                target["estimated_turn"] = item.get("est_turn", target["appointment_time_12h"])
                target["estimated_turn_display"] = target["estimated_turn"]
                target["proj_start"] = item.get("proj_start")
                target["proj_end"] = item.get("proj_end")
                target["delay_min"] = item.get("delay_min", 0.0)
            else:
                target["queue_position"] = "#1"
                target["queue_position_display"] = "#1"
                target["queue_pos_num"] = 1
                target["patients_ahead"] = 0
                target["patients_ahead_display"] = "0"
                target["live_eta_minutes"] = 0.0
                target["live_eta_display"] = "0 minutes"
                target["estimated_turn"] = target["appointment_time_12h"]
                target["estimated_turn_display"] = target["appointment_time_12h"]
            target["is_live_queue"] = True
            target["notifications"] = get_waiting_patient_notifications(target.get("doctor_department", ""), doctor_id) if doctor_id else []

    elif status == "in_service":
        target["queue_position"] = "Current Patient"
        target["queue_position_display"] = "Current Patient"
        target["queue_pos_num"] = 0
        target["patients_ahead"] = 0
        target["patients_ahead_display"] = "0"
        target["live_eta_minutes"] = 0.0
        target["live_eta_display"] = "Now"
        target["estimated_turn"] = "Now"
        target["estimated_turn_display"] = "Now"
        target["is_live_queue"] = True
        target["notifications"] = []
        if item:
            target["proj_start"] = item.get("proj_start")
            target["proj_end"] = item.get("proj_end")
            target["delay_min"] = item.get("delay_min", 0.0)

    elif status == "done":
        target["queue_position"] = "Completed"
        target["queue_position_display"] = "Completed"
        target["queue_pos_num"] = None
        target["patients_ahead"] = None
        target["patients_ahead_display"] = "--"
        target["live_eta_minutes"] = None
        target["live_eta_display"] = "Completed"
        target["estimated_turn"] = "Completed"
        target["estimated_turn_display"] = "Completed"
        target["is_live_queue"] = False
        target["notifications"] = []

    else:
        target["queue_position"] = "N/A"
        target["queue_position_display"] = "N/A"
        target["queue_pos_num"] = None
        target["patients_ahead"] = None
        target["patients_ahead_display"] = "--"
        target["live_eta_minutes"] = None
        target["live_eta_display"] = "N/A"
        target["estimated_turn"] = "N/A"
        target["estimated_turn_display"] = "N/A"
        target["is_live_queue"] = False
        target["notifications"] = []

    return target


def patient_check_in(user: Optional[Dict], booking_id: int) -> None:
    """Self check-in for a patient at the clinic; validates ownership if user is patient."""
    if not user:
        raise PermissionError("Authentication required.")
    booking = store.get_booking(db_path(), int(booking_id))
    if not booking:
        raise ValueError(f"Booking #{booking_id} not found.")
    if user.get("role") == "patient":
        if booking.get("patient_user_id") and int(booking["patient_user_id"]) != int(user.get("id", 0)):
            raise PermissionError("Access denied: cannot check in for another patient.")
    # Date guard: prevent check-in for appointments dated before today.
    appt_date = booking.get("appointment_date", "")
    today_iso = date.today().isoformat()
    if appt_date and appt_date < today_iso:
        raise ValueError(
            "This appointment date has passed. "
            "Check-in is not available for past appointments. "
            "Please book a new appointment."
        )
    store.check_in(db_path(), int(booking_id))


def doctor_check_in(user: Optional[Dict], booking_id: int,
                    predicted_wait: Optional[float] = None) -> None:
    require_role(user, ("doctor", "admin"))
    store.check_in(db_path(), booking_id, predicted_wait=predicted_wait)


def doctor_start_service(user: Optional[Dict], booking_id: int) -> None:
    require_role(user, ("doctor", "admin"))
    store.start_service(db_path(), booking_id)


def doctor_complete_service(user: Optional[Dict], booking_id: int,
                            service_minutes: Optional[float] = None,
                            notes: Optional[str] = None) -> Dict:
    """Complete a consult and auto-reclaim leftover duty time as new ML slots."""
    require_role(user, ("doctor", "admin"))
    store.complete_service(db_path(), booking_id, service_minutes=service_minutes, notes=notes)
    booking = store.get_booking(db_path(), int(booking_id))
    if not booking:
        return {"reclaimed_slots": [], "summary": {}, "time_saved_min": 0.0}
    doc = get_doctor(int(booking["doctor_id"]))
    date_iso = booking.get("appointment_date") or date.today().isoformat()
    schedule = live_schedule(doc, date_iso) if doc else {
        "extra_open_slots": [], "summary": {},
    }
    summary = schedule.get("summary") or {}
    reclaimed = schedule.get("extra_open_slots") or []
    return {
        "reclaimed_slots": reclaimed,
        "summary": summary,
        "time_saved_min": float(summary.get("time_saved_min") or 0.0),
        "finishing_early": bool(summary.get("finishing_early")),
        "new_open_slots": len(reclaimed),
        "suggested_buffer_min": float(summary.get("suggested_buffer_min") or 0.0),
    }


def doctor_set_status(user: Optional[Dict], booking_id: int, status: str) -> None:
    require_role(user, ("doctor", "admin"))
    store.set_status(db_path(), booking_id, status=status)


def assign_emergency_priority(user: Optional[Dict], booking_id: int,
                              priority_level: int = 1,
                              reason: str = "") -> None:
    require_role(user, ("doctor", "admin"))
    actor = f"{user.get('role', 'staff')}:{user.get('username', 'user')}" if user else "staff"
    store.set_emergency_priority(db_path(), booking_id, priority_level=priority_level,
                                 reason=reason, assigned_by=actor)


def evaluate_policies_live(weights: Optional[Dict[str, float]] = None,
                           emergency_level: int = 1) -> Dict:
    """Run real multi-objective policy comparison with custom or default weights."""
    from qms.optimization.objectives import composite_score
    precomputed = policy_comparison()
    default_weights = get_cfg().get("scheduler", {}).get("objective_weights", {
        "waiting": 0.4, "utilization": 0.2, "fairness": 0.2, "emergency": 0.2
    })
    w = weights or default_weights
    if precomputed and "policies" in precomputed:
        policies_data = dict(precomputed["policies"])
        scores = composite_score(policies_data, w)
        for p, s in scores.items():
            if p in policies_data:
                policies_data[p]["composite_score"] = s
        best = max(scores, key=lambda k: scores[k])
        return {
            "weights": w,
            "emergency_level": emergency_level,
            "n_groups": precomputed.get("n_groups", 190),
            "n_patients": precomputed.get("n_patients", 6512),
            "policies": policies_data,
            "best_policy": best,
        }
    return precomputed or {}


def check_booking_conflict(doctor_id: int, appointment_date: str,
                           slot_start: str, slot_end: str,
                           exclude_id: Optional[int] = None) -> bool:
    """Check if (slot_start, slot_end) conflicts with any confirmed booking for the doctor."""
    return store.check_booking_conflict(
        db_path(), int(doctor_id), appointment_date, slot_start, slot_end, exclude_id=exclude_id
    )


def book_appointment(booking_dict: Dict) -> Tuple[Optional[int], Optional[str]]:
    """Validate and book an appointment with conflict checking."""
    seed_default_system()
    rec = dict(booking_dict)
    doc_id = rec.get("doctor_id")
    appt_date = rec.get("appointment_date")
    slot_s = rec.get("slot_start")
    slot_e = rec.get("slot_end")

    if not doc_id:
        return None, "Doctor is required."
    if not appt_date:
        return None, "Appointment date is required."
    if not slot_s:
        return None, "Slot start time is required."

    # Issue 3: Reject patient-side emergency priority requests during booking
    if int(rec.get("priority_level") or 0) >= 1 or rec.get("emergency_reason") or "patient:" in str(rec.get("assigned_by", "")):
        return None, "Emergency priority cannot be requested through standard appointment booking. Emergency walk-ins must be registered by authorized hospital administration."

    # Validate availability on date
    is_avail, reason = is_doctor_available_on_date(int(doc_id), appt_date)
    if not is_avail:
        return None, f"Cannot book appointment: {reason}"

    # Validate backend conflict
    if slot_e and check_booking_conflict(int(doc_id), appt_date, slot_s, slot_e):
        return None, "Cannot book appointment: Selected slot conflicts with an existing appointment for this doctor."

    try:
        booking_id = store.add_booking(db_path(), rec)
        return booking_id, None
    except ValueError as exc:
        return None, str(exc)


def add_emergency_walk_in(admin_user: Dict,
                          patient_name: str,
                          department: str,
                          doctor_id: int,
                          emergency_reason: str,
                          patient_user_id: Optional[int] = None,
                          patient_details: Optional[Dict] = None) -> Tuple[Optional[int], Optional[str]]:
    """Admin-only workflow to insert an emergency walk-in patient directly into the live queue."""
    seed_default_system()
    require_role(admin_user, ("admin",))

    name = str(patient_name or "").strip()
    reason = str(emergency_reason or "").strip()
    dept = str(department or "").strip()
    try:
        doc_id = int(doctor_id)
    except (ValueError, TypeError):
        return None, "Invalid doctor selection."

    if not name:
        return None, "Patient name is required."
    if not reason:
        return None, "Emergency reason is required."
    if not dept:
        return None, "Department is required."

    from datetime import date as _dt_date
    today_iso = _dt_date.today().isoformat()

    # Verify doctor is active and available today
    doc = store.get_doctor(db_path(), doc_id)
    if not doc or not doc.get("active"):
        return None, "Assigned doctor is not currently active."
    if doc.get("department") != dept:
        return None, f"Doctor {doc.get('name')} does not belong to the {dept} department."
    if not is_doctor_available(doc_id, today_iso):
        return None, f"Doctor {doc.get('name')} is not available today ({today_iso})."

    # Duplicate check: check if this patient already has an active check_in or in_service appointment with this doctor today
    existing = store.list_bookings(db_path(), doctor_id=doc_id, date=today_iso,
                                   statuses=["checked_in", "in_service"])
    for ex in existing:
        if ex.get("patient_name", "").strip().lower() == name.lower():
            if patient_user_id is not None and ex.get("patient_user_id") == patient_user_id:
                return None, f"Patient '{name}' is already active in the live queue for Dr. {doc.get('name')}."
            elif patient_user_id is None:
                return None, f"Patient '{name}' is already active in the live queue for Dr. {doc.get('name')}."

    now_m = _now_min()
    now_hhmm = scheduling.to_hhmm(now_m)
    fallback_w = float(doc.get("slot_minutes", 15) or 15)
    end_hhmm = scheduling.to_hhmm(now_m + fallback_w)

    now_iso = datetime.now().isoformat(timespec="seconds")
    actor = f"admin:{admin_user.get('username', 'admin')}"

    booking_rec = {
        "patient_name": name,
        "patient_user_id": patient_user_id,
        "department": dept,
        "doctor_id": doc_id,
        "appointment_date": today_iso,
        "time_slot": "Emergency Walk-In",
        "slot_start": now_hhmm,
        "slot_end": end_hhmm,
        "priority_level": 1,
        "status": "checked_in",
        "check_in_time": now_iso,
        "emergency_reason": reason,
        "assigned_by": actor,
        "created_at": now_iso,
    }

    booking_id = store.add_booking(db_path(), booking_rec)

    # In-app notification for waiting patients in this department/queue:
    # Does NOT include patient name or reason (confidentiality requirement C4)
    notif_msg = (
        "Emergency Alert: An emergency patient has arrived in this department. "
        "Queue positions and estimated waiting times may change. Please follow the latest queue guidance."
    )
    store.create_queue_notification(db_path(), department=dept, doctor_id=doc_id, message=notif_msg)

    return booking_id, None


def get_waiting_patient_notifications(department: str, doctor_id: int) -> List[Dict]:
    """Retrieve today's queue notifications for waiting patients in this department/doctor queue."""
    seed_default_system()
    from datetime import date as _dt_date
    today_iso = _dt_date.today().isoformat()
    return store.list_queue_notifications(db_path(), department=department,
                                         doctor_id=doctor_id, date=today_iso)


def auto_handle_expired_appointments(today_iso: Optional[str] = None,
                                     now_min: Optional[float] = None,
                                     grace_minutes: float = 30.0) -> int:
    """Transition expired booked appointments to 'no_show'."""
    seed_default_system()
    return store.auto_handle_expired_appointments(
        db_path(), today_iso=today_iso, now_min=now_min, grace_minutes=grace_minutes
    )

