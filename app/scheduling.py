"""Doctor-level slot generation and dynamic re-allocation engine.

This is the scheduling brain behind the booking UI. It is deliberately made of
**pure functions** (plain dicts in, plain dicts out — no Streamlit, no database),
so it is easy to unit-test and reuse.

Two jobs
--------
1. **Slot planning.** Given a doctor's duty start/end, pack bookable slots using
   a *predicted* service duration (ML / department history — not a fixed clinic
   width) plus a dynamic buffer. Remaining holes after existing bookings, and
   leftover duty time after the last patient, are packed the same way.

2. **Dynamic re-allocation (live).** As the day runs we know which patients are
   done (and how long they actually took), who is in service, and who has not
   arrived. ``live_schedule`` rolls the plan forward from the real progress:

   * arrival-aware starts — service cannot begin before check-in (or a
     predicted late arrival); “late” on the timeline is clinic delay, not
     patient delay;
   * cascading delays  — if the doctor is running late, later slots slide;
   * leftover refill — if booked work finishes and duty time remains, extra
     slots are generated from predicted service + department wait-aware buffer;
   * no-show reclaim — expected no-shows (from the ML risk scores) are turned
     into a safe *overbooking* headroom via :class:`DynamicBuffer`;
   * adaptive buffer — if today's completed patients are consistently over- or
     under-running, a revised buffer is suggested.

The Live Queue page renders the output of ``live_schedule``; the Book
Appointment page renders the output of ``available_slots``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

DAY_MIN = 24 * 60


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------
def to_min(value) -> Optional[float]:
    """Parse "HH:MM"[:SS], 12-hour AM/PM times (e.g. "10:20 AM"), or an ISO timestamp into minutes-of-day.
    Passes through numbers unchanged. Returns ``None`` for empty or unparseable input."""
    if value is None or value == "" or value == "—":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() == "none":
        return None
    if "T" in s:                     # ISO datetime -> take the time part
        s = s.split("T", 1)[1]
    s = s.split("+", 1)[0].split("Z", 1)[0].strip()
    s_upper = s.upper()
    is_pm = "PM" in s_upper
    is_am = "AM" in s_upper
    try:
        if is_pm or is_am:
            clean = s_upper.replace("AM", "").replace("PM", "").strip()
            parts = clean.split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            sec = float(parts[2]) if len(parts) > 2 else 0.0
            if is_pm and h != 12:
                h += 12
            elif is_am and h == 12:
                h = 0
            return h * 60 + m + sec / 60.0
        parts = s.split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return h * 60 + m + sec / 60.0
    except (ValueError, IndexError):
        return None



def to_hhmm(minutes: float) -> str:
    """Format minutes-of-day as a "HH:MM" clock string."""
    m = int(round(minutes)) % DAY_MIN
    return f"{m // 60:02d}:{m % 60:02d}"


def to_hhmm_12h(minutes: Optional[float]) -> str:
    """Format minutes-of-day into 12-hour AM/PM string, e.g. 600 -> '10:00 AM'."""
    if minutes is None:
        return "—"
    m = int(round(minutes)) % DAY_MIN
    h = m // 60
    mins = m % 60
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:{mins:02d} {ampm}"


def format_clock_12h(value) -> str:
    """Format '10:00' or datetime/minutes to '10:00 AM'."""
    if value is None or value == "" or value == "—":
        return "—"
    m = to_min(value)
    if m is None:
        return str(value)
    return to_hhmm_12h(m)


def format_date_human(date_str: str) -> str:
    """Format '2026-10-04' to 'October 4, 2026'."""
    if not date_str:
        return "—"
    from datetime import date as _dt_date
    try:
        d = _dt_date.fromisoformat(str(date_str).split("T")[0])
        day_num = d.day
        return f"{d.strftime('%B')} {day_num}, {d.year}"
    except Exception:
        return str(date_str)


def format_date_short(date_str: str) -> str:
    """Format '2026-10-04' to '04-10-2026'."""
    if not date_str:
        return "—"
    from datetime import date as _dt_date
    try:
        d = _dt_date.fromisoformat(str(date_str).split("T")[0])
        return d.strftime("%d-%m-%Y")
    except Exception:
        return str(date_str)


# ---------------------------------------------------------------------------
# 1. static slot plan
# ---------------------------------------------------------------------------
def plan_slots(shift_start, shift_end, slot_minutes,
               buffer_minutes: float = 0.0) -> List[Dict]:
    """Pack equal-width slots inside a doctor's shift (used as a primitive).

    Each slot occupies ``slot_minutes`` of service time and is followed by a
    ``buffer_minutes`` gap. A slot is included only if its service portion fits
    entirely before ``shift_end``.
    """
    return generate_shift_slots(
        shift_start, shift_end,
        duration_fn=lambda _t: float(slot_minutes),
        buffer_fn=lambda _t: float(buffer_minutes or 0.0),
    )


def generate_shift_slots(shift_start, shift_end, *,
                         duration_fn: Callable[[float], float],
                         buffer_fn: Callable[[float], float],
                         from_min: Optional[float] = None,
                         occupied: Optional[Sequence[tuple]] = None,
                         origin: str = "planned") -> List[Dict]:
    """Walk the shift and pack slots whose *width* is decided at each start time.

    ``duration_fn(start_min)`` is predicted consult length; ``buffer_fn`` is the
    dynamic gap after that consult. Existing ``occupied`` intervals (start, end)
    are skipped so leftover duty time after booked patients still becomes slots.
    """
    s = to_min(shift_start)
    e = to_min(shift_end)
    slots: List[Dict] = []
    if s is None or e is None:
        return slots
    occ = sorted((float(a), float(b)) for a, b in (occupied or []) if b > a)
    t = s if from_min is None else max(s, float(from_min))
    i = 0
    occ_i = 0
    guard = 0
    while t + 1e-9 < e and guard < 500:
        guard += 1
        while occ_i < len(occ) and occ[occ_i][1] <= t + 1e-9:
            occ_i += 1
        w = float(duration_fn(t) or 0.0)
        b = max(float(buffer_fn(t) or 0.0), 0.0)
        if w <= 0:
            break
        if t + w > e + 1e-9:
            rem = e - t
            if rem >= 5.0:
                w = rem
            else:
                break
        if occ_i < len(occ):
            os, oe = occ[occ_i]
            if t < os and t + w > os + 1e-9:
                t = oe + b
                continue
            if t + 1e-9 < oe and t + w > os:
                t = oe + b
                continue
        slots.append({
            "index": i, "start_min": t, "end_min": t + w,
            "start": to_hhmm(t), "end": to_hhmm(t + w),
            "slot_start": to_hhmm(t), "slot_end": to_hhmm(t + w),
            "service_minutes": round(w, 1), "buffer_minutes": round(b, 1),
            "origin": origin,
        })
        t += w + b
        i += 1
    return slots


def _occupied_intervals(bookings: Sequence[Dict],
                        default_width: float) -> List[tuple]:
    """Reserved windows that a *new* booking must not overlap.

    Completed visits occupy only the *realised* consult, so unused tail of the
    original slot (and leftover duty time after the last finish) can be
    re-offered. No-shows / cancellations free the slot entirely.
    """
    out: List[tuple] = []
    w = float(default_width or 15)
    for b in bookings:
        status = b.get("status")
        if status in ("no_show", "cancelled"):
            continue
        sm = to_min(b.get("slot_start"))
        if sm is None:
            continue
        actual = b.get("service_minutes")
        actual = float(actual) if actual not in (None, "") else None
        em = to_min(b.get("slot_end"))
        if status == "done":
            dur = actual if actual is not None else (em - sm if em is not None else w)
            em = sm + max(dur, 1.0)
        else:
            if em is None:
                em = sm + (actual if actual is not None else w)
            if em <= sm:
                em = sm + w
        if em <= sm:
            em = sm + w
        out.append((sm, em))
    return out


def available_slots(doctor: Dict, bookings: Sequence[Dict],
                    now_min: Optional[float] = None,
                    is_today: bool = False,
                    duration_fn: Optional[Callable[[float], float]] = None,
                    buffer_fn: Optional[Callable[[float], float]] = None,
                    ) -> List[Dict]:
    """Concrete slots a *new* patient can still book with this doctor.

    Packs predicted service + dynamic buffer around already-booked intervals.
    On the current day, time already passed is not offered.
    """
    fallback_w = float(doctor.get("slot_minutes", 15) or 15)
    fallback_b = float(doctor.get("buffer_minutes", 0) or 0)
    dfn = duration_fn or (lambda _t: fallback_w)
    bfn = buffer_fn or (lambda _t: fallback_b)
    shift_end = to_min(doctor.get("shift_end"))
    still_in_shift = (not is_today) or now_min is None or shift_end is None or now_min < shift_end
    from_min = now_min if (is_today and still_in_shift) else None
    occupied = _occupied_intervals(bookings, fallback_w)
    packed = generate_shift_slots(
        doctor.get("shift_start"), doctor.get("shift_end"),
        duration_fn=dfn, buffer_fn=bfn, from_min=from_min,
        occupied=occupied, origin="open")
    return packed


def merge_reclaimed_slots(open_slots: Sequence[Dict], extra_open: Sequence[Dict],
                          now_min: Optional[float] = None,
                          still_in_shift: bool = True) -> List[Dict]:
    """Union of packed open slots and leftover slots from ``live_schedule``.

    A leftover slot stays bookable until it *ends*.
    """
    extra = list(extra_open or [])
    if now_min is not None:
        extra = [s for s in extra if s.get("end_min", s.get("start_min", 0)) > now_min]
        open_slots = [s for s in open_slots
                      if s.get("end_min", s.get("start_min", 0)) > now_min]
    seen = {s["start"] for s in open_slots}
    out = [dict(s) for s in open_slots]
    for s in extra:
        if s.get("start") not in seen:
            slot = dict(s)
            slot.setdefault("origin", "reclaimed")
            out.append(slot)
            seen.add(s["start"])
    for s in out:
        s.setdefault("slot_start", s.get("start"))
        s.setdefault("slot_end", s.get("end"))
    out.sort(key=lambda s: s.get("start_min", 0))
    return out


# ---------------------------------------------------------------------------
_ACTIVE = ("booked", "checked_in", "in_service", "done")


def live_schedule(doctor: Dict, bookings: Sequence[Dict],
                  now_min: Optional[float] = None,
                  max_overbook_fraction: float = 0.30,
                  remaining_service_fn: Optional[Callable[[float], float]] = None,
                  leftover_duration_fn: Optional[Callable[[float], float]] = None,
                  leftover_buffer_fn: Optional[Callable[[float], float]] = None,
                  predicted_wait: float = 0.0,
                  is_today: bool = True,
                  date_iso: Optional[str] = None) -> Dict:
    """Roll the plan forward from real progress and re-allocate remaining time.

    Parameters
    ----------
    doctor : dict with ``shift_start``, ``shift_end``, and fallback
        ``slot_minutes`` / ``buffer_minutes``.
    bookings : list of dicts with ``slot_start`` ("HH:MM"), ``status``,
        ``priority_level``, ``noshow_prob`` and — when available —
        ``service_minutes`` (realised), ``service_start`` / ``check_in_time``
        (ISO), and ``predicted_arrival_delay`` (minutes).
    now_min : current minute-of-day. When omitted, leftover slots are packed
        from the last projected finish (do not assume today's wall clock).
    remaining_service_fn : predicted duration for patients not yet completed.
    leftover_duration_fn / leftover_buffer_fn : how to pack extra slots in
        unused duty time (typically predicted service + wait-aware buffer).
    """
    s = to_min(doctor.get("shift_start"))
    e = to_min(doctor.get("shift_end"))
    w = float(doctor.get("slot_minutes", 15))
    base_buf = float(doctor.get("buffer_minutes", 0) or 0.0)

    def _pred_dur(at: float, b: Optional[Dict] = None) -> float:
        if remaining_service_fn is None:
            return w
        try:
            val = remaining_service_fn(at, b)
        except TypeError:
            val = remaining_service_fn(at)
        val = float(val or w)
        return val if val > 0 else w

    def _gap_after(at: float) -> float:
        """Dynamic inter-patient gap (crowd-aware buffer when provided)."""
        if leftover_buffer_fn is not None:
            try:
                return max(float(leftover_buffer_fn(at) or 0.0), 0.0)
            except Exception:  # noqa: BLE001
                pass
        return max(base_buf, 0.0)

    scheduled_raw = [b for b in bookings if b.get("status") in _ACTIVE]

    # Priority sequencing respecting active consultations and emergency preemption:
    # 1. 'done' visits in completion / planned order
    # 2. 'in_service' visit (current active examination)
    # 3. 'checked_in' emergency patients (priority >= 1) - preempt pending routine patients
    # 4. 'checked_in' routine patients (priority < 1)
    # 5. 'booked' patients (future arrivals, sorted by planned slot)
    done_items = [b for b in scheduled_raw if b.get("status") == "done"]
    done_items.sort(key=lambda b: (to_min(b.get("service_start")) if to_min(b.get("service_start")) is not None else 0,
                                   to_min(b.get("slot_start")) or 0,
                                   b.get("id", 0)))

    in_service_items = [b for b in scheduled_raw if b.get("status") == "in_service"]

    ci_emergencies = [b for b in scheduled_raw if b.get("status") == "checked_in" and int(b.get("priority_level") or 0) >= 1]
    ci_emergencies.sort(key=lambda b: (to_min(b.get("check_in_time")) if to_min(b.get("check_in_time")) is not None else 0,
                                       to_min(b.get("slot_start")) or 0,
                                       b.get("id", 0)))

    ci_routines = [b for b in scheduled_raw if b.get("status") == "checked_in" and int(b.get("priority_level") or 0) < 1]
    ci_routines.sort(key=lambda b: (to_min(b.get("slot_start")) if to_min(b.get("slot_start")) is not None else 9999,
                                    to_min(b.get("check_in_time")) if to_min(b.get("check_in_time")) is not None else 0,
                                    b.get("id", 0)))

    checked_in_ordered = ci_emergencies + ci_routines

    booked_items = [b for b in scheduled_raw if b.get("status") == "booked"]
    booked_items.sort(key=lambda b: (to_min(b.get("slot_start")) if to_min(b.get("slot_start")) is not None else (e or 9999),
                                     -int(b.get("priority_level") or 0),
                                     b.get("id", 0)))

    scheduled = done_items + in_service_items + checked_in_ordered + booked_items

    timeline: List[Dict] = []
    overruns: List[float] = []
    cursor = s if s is not None else (now_min or 0.0)
    last_applied_gap = 0.0

    shift_ended = bool(is_today and now_min is not None and e is not None and now_min >= e)
    # Effective current time within shift (do not project intra-shift queue into off-shift hours)
    effective_now = min(now_min, e) if (is_today and now_min is not None and e is not None) else (now_min if is_today else None)

    # 1. Process completed visits (done_items)
    for b in done_items:
        planned = to_min(b.get("slot_start"))
        planned = planned if planned is not None else cursor
        actual = b.get("service_minutes")
        actual = float(actual) if actual not in (None, "") else None
        dur = actual if actual is not None else _pred_dur(planned, b)
        overruns.append(dur - _pred_dur(planned, b))
        started = to_min(b.get("service_start")) or to_min(b.get("check_in_time"))
        if started is not None:
            proj_start = started
        else:
            ready = _ready_min(b, planned)
            if is_today and effective_now is not None and planned > effective_now:
                proj_start = cursor
            else:
                proj_start = max(cursor, ready)
        proj_end = proj_start + dur
        last_applied_gap = _gap_after(proj_end)
        cursor = max(cursor, proj_end + last_applied_gap)
        arrived = to_min(b.get("check_in_time"))
        arrival_delay = (arrived - planned) if arrived is not None else _predicted_late_min(b)
        timeline.append({
            "id": b.get("id"),
            "patient_name": b.get("patient_name"),
            "priority_level": int(b.get("priority_level") or 0),
            "emergency_reason": b.get("emergency_reason") or "",
            "assigned_by": b.get("assigned_by") or "",
            "notes": b.get("notes") or "",
            "status": "done",
            "queue_position": "Completed",
            "queue_pos_num": None,
            "patients_ahead": 0,
            "est_wait_min": None,
            "est_turn": "Completed",
            "noshow_prob": b.get("noshow_prob"),
            "planned_start": to_hhmm(planned),
            "proj_start": to_hhmm(proj_start),
            "proj_end": to_hhmm(proj_end),
            "proj_start_min": round(proj_start, 1),
            "proj_end_min": round(proj_end, 1),
            "delay_min": round(proj_start - planned, 1),
            "arrival_delay_min": round(arrival_delay, 1) if arrival_delay is not None else None,
            "service_min": round(dur, 1),
        })

    if not in_service_items and s is not None:
        cursor = max(cursor, s)

    # 2. Process in-service items (current active consultation)
    for b in in_service_items:
        planned = to_min(b.get("slot_start"))
        planned = planned if planned is not None else cursor
        actual = b.get("service_minutes")
        actual = float(actual) if actual not in (None, "") else None
        ready = _ready_min(b, planned)
        started = to_min(b.get("service_start"))
        proj_start = started if started is not None else max(cursor, ready)
        dur = actual if actual is not None else _pred_dur(proj_start, b)
        if effective_now is not None and started is not None:
            elapsed = max(0.0, effective_now - started)
            rem_dur = max(dur - elapsed, 2.0)
            proj_end = effective_now + rem_dur
        else:
            proj_end = proj_start + dur
            if effective_now is not None:
                proj_end = max(proj_end, effective_now)
        last_applied_gap = _gap_after(proj_end)
        cursor = proj_end + last_applied_gap
        arrived = to_min(b.get("check_in_time"))
        arrival_delay = (arrived - planned) if arrived is not None else _predicted_late_min(b)
        timeline.append({
            "id": b.get("id"),
            "patient_name": b.get("patient_name"),
            "priority_level": int(b.get("priority_level") or 0),
            "emergency_reason": b.get("emergency_reason") or "",
            "assigned_by": b.get("assigned_by") or "",
            "notes": b.get("notes") or "",
            "status": "in_service",
            "queue_position": "Current Patient",
            "queue_pos_num": 0,
            "patients_ahead": 0,
            "est_wait_min": 0.0,
            "est_turn": "Now",
            "noshow_prob": b.get("noshow_prob"),
            "planned_start": to_hhmm(planned),
            "proj_start": to_hhmm(proj_start),
            "proj_end": to_hhmm(proj_end),
            "proj_start_min": round(proj_start, 1),
            "proj_end_min": round(proj_end, 1),
            "delay_min": round(proj_start - planned, 1),
            "arrival_delay_min": round(arrival_delay, 1) if arrival_delay is not None else None,
            "service_min": round(dur, 1),
        })

    # 3. Process checked-in waiting patients
    for ci_idx, b in enumerate(checked_in_ordered):
        planned = to_min(b.get("slot_start"))
        planned = planned if planned is not None else cursor
        ready = _ready_min(b, planned)
        ahead = ci_idx + len(in_service_items)
        pos_num = ahead + 1

        proj_start = max(cursor, ready)
        dur = _pred_dur(proj_start, b)
        proj_end = proj_start + dur
        last_applied_gap = _gap_after(proj_end)
        cursor = proj_end + last_applied_gap

        queue_pos_display = f"#{pos_num}"
        queue_pos_num = pos_num
        patients_ahead = ahead

        if ahead == 0 and not in_service_items:
            est_wait_min = 0.0
        else:
            ref_time = effective_now if (effective_now is not None and is_today) else (s or 0.0)
            wait_min = max(0.0, proj_start - ref_time)
            est_wait_min = round(wait_min, 1)

        if shift_ended:
            est_turn_display = f"Shift Concluded ({to_hhmm_12h(e)})"
            est_wait_min = None
        elif e is not None and proj_start > e:
            est_turn_display = f"{to_hhmm_12h(proj_start)} (Exceeds shift end {to_hhmm_12h(e)})"
        else:
            est_turn_display = to_hhmm_12h(proj_start)

        arrived = to_min(b.get("check_in_time"))
        arrival_delay = (arrived - planned) if arrived is not None else _predicted_late_min(b)
        timeline.append({
            "id": b.get("id"),
            "patient_name": b.get("patient_name"),
            "priority_level": int(b.get("priority_level") or 0),
            "emergency_reason": b.get("emergency_reason") or "",
            "assigned_by": b.get("assigned_by") or "",
            "notes": b.get("notes") or "",
            "status": "checked_in",
            "queue_position": queue_pos_display,
            "queue_pos_num": queue_pos_num,
            "patients_ahead": patients_ahead,
            "est_wait_min": est_wait_min,
            "est_turn": est_turn_display,
            "noshow_prob": b.get("noshow_prob"),
            "planned_start": to_hhmm(planned),
            "proj_start": to_hhmm(proj_start),
            "proj_end": to_hhmm(proj_end),
            "proj_start_min": round(proj_start, 1),
            "proj_end_min": round(proj_end, 1),
            "delay_min": round(proj_start - planned, 1),
            "arrival_delay_min": round(arrival_delay, 1) if arrival_delay is not None else None,
            "service_min": round(dur, 1),
        })

    # 4. Process booked items (future arrivals not yet checked in)
    for b in booked_items:
        planned = to_min(b.get("slot_start"))
        planned = planned if planned is not None else cursor
        ready = _ready_min(b, planned)
        dur = _pred_dur(planned, b)

        if is_today and effective_now is not None:
            proj_start = max(cursor, ready, effective_now)
        else:
            proj_start = max(cursor, ready)
        proj_end = proj_start + dur
        last_applied_gap = _gap_after(proj_end)
        cursor = proj_end + last_applied_gap
        queue_pos_display = "Available after check-in"
        queue_pos_num = None
        patients_ahead = None
        est_wait_min = None

        if e is not None and planned is not None and planned > e:
            est_turn_display = f"{to_hhmm_12h(planned)} (Outside shift {to_hhmm_12h(e)})"
        else:
            est_turn_display = to_hhmm_12h(planned) if planned is not None else "—"

        arrived = to_min(b.get("check_in_time"))
        arrival_delay = (arrived - planned) if arrived is not None else _predicted_late_min(b)
        timeline.append({
            "id": b.get("id"),
            "patient_name": b.get("patient_name"),
            "priority_level": int(b.get("priority_level") or 0),
            "emergency_reason": b.get("emergency_reason") or "",
            "assigned_by": b.get("assigned_by") or "",
            "notes": b.get("notes") or "",
            "status": "booked",
            "queue_position": queue_pos_display,
            "queue_pos_num": queue_pos_num,
            "patients_ahead": patients_ahead,
            "est_wait_min": est_wait_min,
            "est_turn": est_turn_display,
            "noshow_prob": b.get("noshow_prob"),
            "planned_start": to_hhmm(planned),
            "proj_start": to_hhmm(proj_start),
            "proj_end": to_hhmm(proj_end),
            "proj_start_min": round(proj_start, 1),
            "proj_end_min": round(proj_end, 1),
            "delay_min": round(proj_start - planned, 1),
            "arrival_delay_min": round(arrival_delay, 1) if arrival_delay is not None else None,
            "service_min": round(dur, 1),
        })

    # ---- adaptive buffer: no-show + overrun + crowd pressure ------------
    pending = [b for b in scheduled if b.get("status") in ("booked", "checked_in")]
    waiting_n = len(checked_in_ordered)
    pending_n = len(pending)
    probs = [float(b["noshow_prob"]) for b in pending
             if b.get("noshow_prob") not in (None, "")]
    overbook = _overbook_headroom(probs, max_overbook_fraction)

    if overruns:
        mean_overrun = sum(overruns) / len(overruns)
        suggested_buffer = max(0.0, round(base_buf + mean_overrun, 1))
    else:
        mean_overrun = 0.0
        suggested_buffer = base_buf
    # Provisional slack so crowd/behind compression can react to the day pace
    provisional_slack = ((e - cursor) if e is not None else 0.0)
    try:
        from qms.optimization.buffer import DynamicBuffer
        # Keep doctor fallback at 0 when explicitly configured; service layer
        # supplies the config base (typically 10) via leftover_buffer_fn.
        suggested_buffer = DynamicBuffer(
            base_minutes=max(float(base_buf), 0.0),
            max_overbook_fraction=max_overbook_fraction,
        ).slot_buffer(
            predicted_wait=predicted_wait,
            mean_overrun=mean_overrun,
            noshow_probs=probs,
            waiting_count=waiting_n,
            pending_count=pending_n,
            slack_min=provisional_slack,
            min_minutes=0.0 if base_buf <= 0 else 1.0,
            max_minutes=20.0,
        )
    except Exception:  # noqa: BLE001
        pass

    # ---- leftover duty time -> extra slots (predicted service + buffer) --
    def _adapt_w_fn(raw_fn):
        if raw_fn is None:
            return lambda _t: w
        def _adapted(t: float) -> float:
            try:
                val = raw_fn(t, None)
            except TypeError:
                val = raw_fn(t)
            return float(val or w)
        return _adapted

    extra_w_fn = _adapt_w_fn(leftover_duration_fn or remaining_service_fn)
    extra_b_fn = leftover_buffer_fn or (lambda _t: suggested_buffer)
    active_scheduled = in_service_items + checked_in_ordered + booked_items

    # If all scheduled appointments are done (no active patient in queue) and shift is ongoing:
    # Doctor is available from max(last_done_end, effective_now) until shift_end.
    if not active_scheduled and is_today and effective_now is not None and not shift_ended:
        last_done = max((item["proj_end_min"] for item in timeline), default=(s or 0.0))
        leftover_from = max(s or 0.0, effective_now, last_done)
    else:
        leftover_from = cursor
        if effective_now is not None:
            leftover_from = max(cursor, effective_now)

    extra_open = generate_shift_slots(
        doctor.get("shift_start"), doctor.get("shift_end"),
        duration_fn=extra_w_fn, buffer_fn=extra_b_fn,
        from_min=leftover_from, occupied=[], origin="reclaimed")

    sample_w = float(extra_w_fn(cursor) or w)
    sample_step = sample_w + float(extra_b_fn(cursor) or 0.0)

    # Projected finish represents when the active workload completes
    if shift_ended:
        if done_items:
            proj_finish_val = max((item["proj_end_min"] for item in timeline if item["status"] == "done"), default=(e or 0.0))
        else:
            proj_finish_val = (cursor - last_applied_gap) if last_applied_gap > 0 else cursor
        slack = (e - proj_finish_val) if e is not None else 0.0
        projected_finish_str = to_hhmm(proj_finish_val)
        projected_finish_12h_str = f"Shift Concluded ({to_hhmm_12h(e)})"
        finishing_early = False
    elif active_scheduled:
        proj_finish_val = (cursor - last_applied_gap) if last_applied_gap > 0 else cursor
        slack = (e - proj_finish_val) if e is not None else 0.0
        projected_finish_str = to_hhmm(proj_finish_val)
        projected_finish_12h_str = to_hhmm_12h(proj_finish_val)
        finishing_early = slack >= sample_step
    elif done_items:
        # All patients completed, doctor is free for remainder of shift
        last_done_end = max((item["proj_end_min"] for item in timeline), default=(s or 0.0))
        proj_finish_val = last_done_end
        slack = (e - proj_finish_val) if e is not None else 0.0
        projected_finish_str = to_hhmm(proj_finish_val)
        projected_finish_12h_str = to_hhmm_12h(proj_finish_val)
        finishing_early = slack >= sample_step
    else:
        proj_finish_val = s if s is not None else (effective_now or 540.0)
        slack = (e - proj_finish_val) if e is not None else 0.0
        projected_finish_str = to_hhmm(proj_finish_val)
        projected_finish_12h_str = to_hhmm_12h(proj_finish_val)
        finishing_early = slack >= sample_step

    planned_slots = generate_shift_slots(
        doctor.get("shift_start"), doctor.get("shift_end"),
        duration_fn=_adapt_w_fn(remaining_service_fn),
        buffer_fn=lambda _t: suggested_buffer if leftover_buffer_fn is None else leftover_buffer_fn(_t),
        origin="planned")

    return {
        "doctor": {"id": doctor.get("id"), "name": doctor.get("name"),
                   "department": doctor.get("department"),
                   "shift": f"{to_hhmm(s)}–{to_hhmm(e)}" if s is not None and e is not None else "",
                   "shift_12h": f"{to_hhmm_12h(s)} – {to_hhmm_12h(e)}" if s is not None and e is not None else "",
                   "slot_minutes": w, "buffer_minutes": base_buf},
        "now": to_hhmm(now_min) if now_min is not None else "",
        "now_12h": to_hhmm_12h(now_min) if now_min is not None else "",
        "timeline": timeline,
        "extra_open_slots": extra_open,
        "summary": {
            "planned_slot_count": len(planned_slots),
            "booked_count": len(booked_items),
            "checked_in_count": len(checked_in_ordered),
            "in_service_count": len(in_service_items),
            "done_count": len(done_items),
            "total_scheduled": len(scheduled),
            "projected_finish": projected_finish_str,
            "projected_finish_12h": projected_finish_12h_str,
            "shift_end": to_hhmm(e) if e is not None else "",
            "shift_end_12h": to_hhmm_12h(e) if e is not None else "",
            "slack_min": round(slack, 1),
            "finishing_early": bool(finishing_early),
            "new_open_slots": len(extra_open),
            "expected_no_shows": round(sum(probs), 2) if probs else 0.0,
            "safe_overbook": overbook,
            "mean_overrun_min": round(mean_overrun, 1),
            "time_saved_min": round(max(0.0, -sum(overruns)), 1) if overruns else 0.0,
            "emergency_count": sum(1 for b in scheduled if int(b.get("priority_level") or 0) >= 1),
            "suggested_buffer_min": suggested_buffer,
            "predicted_service_min": round(sample_w, 1),
            "dept_avg_wait_min": round(float(predicted_wait or 0.0), 1),
            "waiting_count": waiting_n,
            "buffer_mode": "dynamic_crowd_aware",
        },
    }


def _predicted_late_min(booking: Dict) -> Optional[float]:
    pred = booking.get("predicted_arrival_delay")
    if pred in (None, ""):
        return None
    try:
        return float(pred)
    except (TypeError, ValueError):
        return None


def _ready_min(booking: Dict, planned: float) -> float:
    """Earliest minute the patient can start: not before they are present.

    Emergency patients can start as soon as they have checked in.
    Routine checked-in patients use actual check-in (not before their booked slot).
    Booked patients slide later by predicted *late* arrival only — predicted
    early does not pull the slot forward.
    """
    arrived = to_min(booking.get("check_in_time"))
    if arrived is not None:
        if int(booking.get("priority_level") or 0) >= 1:
            return arrived
        return max(planned, arrived)
    pred = _predicted_late_min(booking)
    if pred is None:
        return planned
    return planned + max(pred, 0.0)


def _overbook_headroom(probs: Sequence[float], max_fraction: float) -> int:
    """Expected no-shows, capped by a safety fraction. Uses the project's
    :class:`DynamicBuffer` when importable, else replicates its rule."""
    n = len(probs)
    if n == 0:
        return 0
    try:
        from qms.optimization.buffer import DynamicBuffer
        rec = DynamicBuffer(max_overbook_fraction=max_fraction).recommend(probs)
        return int(rec["recommended_overbook"])
    except Exception:  # noqa: BLE001
        import math
        expected = sum(probs)
        cap = int(math.floor(max_fraction * n))
        return int(min(math.floor(expected), cap))
