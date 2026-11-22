"""Hospital administrator portal — clean, real-time, AI-integrated."""

from __future__ import annotations

from datetime import date, datetime

import streamlit as st
import service
import store


def _logout():
    st.session_state["user"] = None
    st.session_state.pop("portal", None)
    st.rerun()


# ---------------------------------------------------------------------------
# Emergency patient section
# ---------------------------------------------------------------------------

def _add_emergency_patient_section(user):
    st.markdown("### 🚨 Add Emergency Patient to Live Queue")
    st.caption(
        "Admin-authorized clinical triage. Directly inserts an urgent walk-in patient "
        "into the department's live queue ahead of routine waiting patients."
    )

    db = service.db_path()
    today_iso = date.today().isoformat()

    with st.expander("➕ Open Emergency Walk-In Registration Form", expanded=True):
        col_type, col_dept = st.columns(2)
        with col_type:
            patient_type = st.radio(
                "Patient Type",
                ["Existing Registered Patient", "New Walk-In Emergency Patient"],
                horizontal=True,
                key="em_patient_type",
            )
        with col_dept:
            departments = service.specialty_departments()
            selected_dept = st.selectbox("Department", departments, key="em_department_select")

        patient_user_id = None
        patient_name = ""

        if patient_type == "Existing Registered Patient":
            patients = store.list_users(db, role="patient")
            if not patients:
                st.warning("No registered patient accounts found.")
            else:
                p_options = {p["id"]: f"{p['full_name']} (@{p['username']})" for p in patients}
                sel_pid = st.selectbox(
                    "Select Registered Patient",
                    list(p_options.keys()),
                    format_func=lambda pid: p_options[pid],
                    key="em_existing_patient_select",
                )
                patient_user_id = sel_pid
                patient_name = next(p["full_name"] for p in patients if p["id"] == sel_pid)
        else:
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                patient_name = st.text_input("Patient Full Name", placeholder="e.g. Rahul Sharma", key="em_walkin_name")
            with col_n2:
                st.text_input("Contact Phone (optional)", placeholder="+91-9999-000000", key="em_walkin_phone")

        all_dept_docs = service.list_doctors(department=selected_dept, active_only=True)
        avail_dept_docs = [d for d in all_dept_docs if service.is_doctor_available(d["id"], today_iso)]

        if not avail_dept_docs:
            st.error(f"No active doctors available in {selected_dept} today.")
            doctor_id = None
        else:
            doc_options = {d["id"]: f"{d['name']} (Duty: {d['shift_start']}–{d['shift_end']})" for d in avail_dept_docs}
            doctor_id = st.selectbox(
                f"Assign Available Doctor ({selected_dept})",
                list(doc_options.keys()),
                format_func=lambda did: doc_options[did],
                key="em_doctor_select",
            )

        emergency_reason = st.text_area(
            "Emergency Reason / Clinical Urgency *",
            placeholder="e.g. Acute chest pain, respiratory distress, trauma",
            key="em_reason_text",
        )

        if st.button("🚨 Add Emergency Patient to Live Queue", type="primary", use_container_width=True, key="btn_submit_emergency_walkin"):
            if not patient_name or not patient_name.strip():
                st.error("Please provide the patient name.")
                return
            if not doctor_id:
                st.error(f"No available doctor in {selected_dept} today.")
                return
            if not emergency_reason or not emergency_reason.strip():
                st.error("Please provide an emergency reason.")
                return

            current_user = user or st.session_state.get("user")
            booking_id, err = service.add_emergency_walk_in(
                admin_user=current_user,
                patient_name=patient_name.strip(),
                department=selected_dept,
                doctor_id=doctor_id,
                emergency_reason=emergency_reason.strip(),
                patient_user_id=patient_user_id,
            )

            if err:
                st.error(err)
            else:
                st.success(
                    f"✅ Emergency patient '{patient_name.strip()}' added to {selected_dept} live queue! "
                    f"(Record ID: #{booking_id}, Priority: Emergency)"
                )
                st.rerun()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _dashboard(user=None):
    if user is None:
        user = st.session_state.get("user")

    st.subheader("📊 Hospital Live Dashboard")

    # Auto-refresh every 30 seconds
    st.caption(f"🔴 Live — last refreshed at {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Now", key="admin_dash_refresh"):
        st.rerun()

    db = service.db_path()

    col_date1, col_date2 = st.columns([2, 3])
    with col_date1:
        selected_date = st.date_input(
            "Select Dashboard Date",
            value=date.today(),
            key="admin_dash_queue_date",
        )
    with col_date2:
        st.write("")
        view_all_dates = st.checkbox("Show All-Time Cumulative Metrics", value=False, key="admin_dash_view_all")

    target_date_iso = None if view_all_dates else selected_date.isoformat()
    sel_date_formatted = "All Dates" if view_all_dates else selected_date.strftime("%d-%m-%Y")

    all_doctors = service.list_all_hospital_doctors()
    active_doctors = [d for d in all_doctors if d.get("active")]

    if view_all_dates:
        active_and_available = active_doctors
    else:
        active_and_available = [
            d for d in active_doctors
            if service.is_doctor_available(d["id"], selected_date.isoformat())
        ]

    patients = store.list_users(db, role="patient")
    appointments = store.list_bookings(db, date=target_date_iso)
    counts = store.counts_by_status(db, date=target_date_iso)

    # Metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Active Doctors", len(active_and_available),
                  help=f"Doctors on duty today (Total roster: {len(all_doctors)})")
    with c2:
        st.metric("Registered Patients", len(patients))
    with c3:
        st.metric(f"Total Appointments", len(appointments))
    with c4:
        st.metric("Checked In", counts.get("checked_in", 0), help="Waiting in queue")
    with c5:
        st.metric("In Consultation", counts.get("in_service", 0), help="Currently seeing doctor")
    with c6:
        st.metric("Completed", counts.get("done", 0), help="Consultations completed")

    st.divider()

    # Appointment status breakdown
    st.markdown(f"### 📋 Appointment Status — {sel_date_formatted}")
    status_order = ["booked", "checked_in", "in_service", "done", "no_show", "cancelled"]
    rows = [
        {
            "Status": "Completed" if k == "done" else "In Consultation" if k == "in_service" else k.replace("_", " ").title(),
            "Count": counts.get(k, 0),
        }
        for k in status_order if k in counts
    ]
    for k, v in counts.items():
        if k not in status_order:
            rows.append({"Status": k.replace("_", " ").title(), "Count": v})

    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info(f"No appointments found for {sel_date_formatted}.")

    st.divider()

    # Emergency walk-in
    _add_emergency_patient_section(user)

    st.divider()

    # Live doctor queue
    st.markdown(f"### 👨‍⚕️ Live Doctor Queue — {sel_date_formatted}")
    st.caption("Doctors automatically appear here once they publish duty hours for the day.")

    col_filter1, _ = st.columns([2, 2])
    with col_filter1:
        avail_names = ["All Available Doctors"] + [d["name"] for d in active_and_available]
        selected_doc = st.selectbox(
            f"Filter by Doctor ({sel_date_formatted})",
            avail_names,
            key="admin_dash_selected_doc",
        )

    sel_date_iso = selected_date.isoformat()
    doctors_to_show = active_and_available
    if selected_doc != "All Available Doctors":
        doctors_to_show = [d for d in active_and_available if d["name"] == selected_doc]

    if not active_and_available:
        st.info(
            f"No doctors have published duty hours for {sel_date_formatted} yet. "
            "Doctors open their slots from the Doctor portal (Shift & Timetable tab)."
        )
    elif not doctors_to_show:
        st.info("Selected doctor is not on duty on this date.")
    else:
        lifecycle_display = {
            "booked": "📅 Booked",
            "checked_in": "✅ Checked In",
            "in_service": "🩺 In Consultation",
            "done": "✔️ Completed",
            "no_show": "⚠️ No Show",
            "cancelled": "❌ Cancelled",
        }

        for doc in doctors_to_show:
            doc_appts = store.list_bookings(db, doctor_id=doc["id"], date=sel_date_iso)
            patient_count = len(doc_appts)
            patient_label = f"{patient_count} {'Patient' if patient_count == 1 else 'Patients'}"
            active_p = next((a for a in doc_appts if a.get("status") == "in_service"), None)
            active_tag = f" — 🩺 Seeing: {active_p['patient_name']}" if active_p else ""
            expander_title = f"👨‍⚕️ {doc['name']} — {doc['department']} — {patient_label}{active_tag}"

            with st.expander(expander_title, expanded=(patient_count > 0 or bool(active_p))):
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.markdown(f"**Doctor:** {doc['name']}")
                    st.markdown(f"**Department:** {doc['department']}")
                with dc2:
                    st.markdown(f"**Shift:** {doc['shift_start']} – {doc['shift_end']}")
                    st.markdown("**Duty Status:** 🟢 On Duty")
                with dc3:
                    st.markdown(f"**Patients Today:** {patient_count}")
                    if active_p:
                        st.markdown(f"**Current Patient:** 🩺 {active_p['patient_name']}")

                if active_p:
                    st.info(f"🩺 **Active Consultation in Progress:** {active_p['patient_name']} (Department: {doc['department']})")

                if patient_count == 0:
                    st.info("No patients assigned yet today.")
                else:
                    for idx, appt in enumerate(doc_appts, start=1):
                        p_name = appt.get("patient_name", "Unknown Patient")
                        raw_status = appt.get("status", "booked")
                        status_str = lifecycle_display.get(raw_status, raw_status.replace("_", " ").title())
                        slot_str = f" · Slot: {appt['slot_start']}" if appt.get("slot_start") else ""
                        prio_str = " · 🚨 **Emergency**" if appt.get("priority_level", 0) >= 1 else ""
                        st.markdown(f"**{idx}. {p_name}** — `{status_str}`{slot_str}{prio_str}")


# ---------------------------------------------------------------------------
# Doctor management (no manual status toggle — doctors auto-appear via duty)
# ---------------------------------------------------------------------------

def _doctor_management():
    st.subheader("👨‍⚕️ Doctor & Hospital Roster")

    db = service.db_path()
    doctors = service.list_all_hospital_doctors()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Doctors", len(doctors))
    with m2:
        st.metric("Active", sum(1 for d in doctors if d.get("active")))
    with m3:
        st.metric("Inactive / Off Roster", sum(1 for d in doctors if not d.get("active")))

    # --- Full roster table ---
    st.markdown("### 📋 All Hospital Doctors")
    st.caption("Doctor availability is managed by each doctor from their portal. Doctors appear in the live queue automatically when they publish duty hours.")

    if doctors:
        st.dataframe(
            [
                {
                    "Doctor ID": d["id"],
                    "Name": d["name"],
                    "Department": d["department"],
                    "Status": "Active" if d.get("active") else "Inactive",
                    "Default Shift Start": d.get("shift_start", "09:00"),
                    "Default Shift End": d.get("shift_end", "13:00"),
                }
                for d in doctors
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.info(
        "💡 Doctors publish their own duty hours from the **Doctor Portal → Shift & Timetable** tab. "
        "Slots are auto-generated by the ML service-time model — admin does not control daily availability."
    )

    # --- On-duty today (read-only) ---
    st.divider()
    st.markdown("### 🟢 Doctors On Duty Today (Auto-Published by Doctors)")
    today_iso = date.today().isoformat()
    on_duty = service.list_doctor_availability(available_date=today_iso)
    on_duty = [r for r in on_duty if r.get("is_available")]
    if on_duty:
        st.dataframe(
            [
                {
                    "Doctor": r.get("doctor_name"),
                    "Department": r.get("department"),
                    "Shift": f"{r.get('shift_start') or '—'} – {r.get('shift_end') or '—'}",
                    "Published by": r.get("published_by") or "doctor",
                }
                for r in on_duty
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No doctors have published duty hours for today yet.")

    # --- Login credentials ---
    st.divider()
    st.markdown("### 🔐 Doctor Login Credentials")
    if doctors:
        st.dataframe(
            [
                {
                    "Doctor": d["name"],
                    "Department": d["department"],
                    "Username": d.get("username") or "—",
                    "Password": d.get("login_password") or "doctor123",
                }
                for d in doctors
            ],
            hide_index=True,
            use_container_width=True,
        )

    # --- Add doctor ---
    st.divider()
    st.markdown("### ➕ Add Doctor")
    st.caption("Register a new doctor and create their login account.")

    departments = service.specialty_departments()

    name = st.text_input("Doctor name", placeholder="e.g. Dr. Ramesh Gupta", key="admin_doc_name")
    department = st.selectbox("Department", departments, key="admin_doc_department")

    c1, c2 = st.columns(2)
    with c1:
        shift_start = st.text_input("Default shift start", value="09:00", key="admin_doc_shift_start")
    with c2:
        shift_end = st.text_input("Default shift end", value="13:00", key="admin_doc_shift_end")

    default_uname = ""
    if name.strip():
        clean_n = name.strip().lower().replace("dr.", "").replace("dr", "").strip()
        default_uname = "dr_" + "_".join(clean_n.split())

    c3, c4 = st.columns(2)
    with c3:
        username = st.text_input("Username", value=default_uname, placeholder="e.g. dr_gupta", key="admin_doc_username")
    with c4:
        password = st.text_input("Login Password", value="doctor123", key="admin_doc_password")

    active = st.checkbox("Active doctor", value=True, key="admin_doc_active")

    if st.button("Add Doctor", type="primary", key="btn_add_doctor"):
        if not name.strip():
            st.error("Enter the doctor's name.")
            return
        if not username.strip():
            st.error("Enter a username.")
            return
        if not password or len(password) < 4:
            st.error("Password must be at least 4 characters.")
            return

        try:
            doctor_id, err = service.add_doctor_with_login(
                {
                    "name": name.strip(),
                    "department": department,
                    "shift_start": shift_start,
                    "shift_end": shift_end,
                    "slot_minutes": 0,
                    "buffer_minutes": float(service.get_cfg().get("buffer", {}).get("base_minutes", 10)),
                    "active": 1 if active else 0,
                },
                username=username.strip(),
                password=password,
            )

            if err:
                st.error(err)
                return

            st.success(
                f"✅ Doctor created! ID: {doctor_id}, Username: {username.strip()}. "
                "They must publish duty hours from the Doctor portal to open bookable slots."
            )
            st.rerun()

        except Exception as exc:
            st.error(f"Could not add doctor: {exc}")


# ---------------------------------------------------------------------------
# Patient management
# ---------------------------------------------------------------------------

def _patient_management():
    st.subheader("👥 Patient Accounts")

    patients = store.list_users(service.db_path(), role="patient")

    if not patients:
        st.info("No patient accounts found.")
        return

    st.dataframe(
        [
            {
                "ID": p["id"],
                "Name": p["full_name"],
                "Username": p["username"],
                "Email": p.get("email"),
                "Phone": p.get("phone"),
                "Registered": p.get("created_at"),
            }
            for p in patients
        ],
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Appointment management
# ---------------------------------------------------------------------------

def _appointments():
    st.subheader("📅 Appointment Management")

    db = service.db_path()
    selected_date = st.date_input("Appointment date", value=date.today(), key="admin_appt_date")
    bookings = store.list_bookings(db, date=selected_date.isoformat())

    if not bookings:
        st.info("No appointments found for the selected date.")
        return

    rows = []
    for b in bookings:
        rows.append(
            {
                "ID": b["id"],
                "Patient": b["patient_name"],
                "Department": b["department"],
                "Doctor ID": b["doctor_id"],
                "Slot": b["slot_start"],
                "Priority": "EMERGENCY" if b.get("priority_level", 0) >= 1 else "Routine",
                "Status": b["status"].replace("_", " ").title(),
                "No-show Prob": (
                    f"{float(b['noshow_prob']) * 100:.1f}%"
                    if b.get("noshow_prob") is not None
                    else "—"
                ),
            }
        )

    st.dataframe(rows, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# AI Copilot — real-time hospital context, answers any question
# ---------------------------------------------------------------------------

def _clinical_copilot():
    st.subheader("🧠 Hospital AI Copilot")
    st.caption(
        "Ask anything about today's operations — queue status, doctor availability, "
        "appointment counts, patient flow, no-show risk, slot capacity, or department load. "
        "Answers are grounded in live hospital data."
    )

    if "admin_chat" not in st.session_state:
        st.session_state["admin_chat"] = []

    c_hist, c_clear = st.columns([5, 1])
    with c_clear:
        if st.button("🗑️ Clear Chat", key="admin_clear_chat"):
            st.session_state["admin_chat"] = []
            st.rerun()

    clicked_question = None
    # Suggested prompts
    with st.expander("💡 Suggested questions", expanded=False):
        suggestions = [
            "How many patients are currently waiting across all departments?",
            "Which doctors are on duty today and how many patients do they each have?",
            "What is the total appointment count today and how many are completed?",
            "Which department has the highest load right now?",
            "How many patients are checked in but not yet seen?",
            "What is the no-show rate today?",
            "Which doctor has the most patients remaining?",
            "Give me a summary of today's hospital operations.",
        ]
        for s in suggestions:
            if st.button(s, key=f"sugg_{s[:30]}"):
                clicked_question = s

    # Chat history
    for message in st.session_state["admin_chat"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    typed_question = st.chat_input("Ask about hospital operations, queue, doctors, patients...")
    question = typed_question or clicked_question

    if not question:
        return

    st.session_state["admin_chat"].append({"role": "user", "content": question})

    # Build comprehensive real-time context
    db = service.db_path()
    today = date.today().isoformat()

    all_doctors = service.list_all_hospital_doctors()
    active_doctors = [d for d in all_doctors if d.get("active")]

    # Doctors on duty today
    on_duty_today = [
        d for d in active_doctors
        if service.is_doctor_available(d["id"], today)
    ]

    bookings_today = store.list_bookings(db, date=today)
    counts = store.counts_by_status(db, date=today)
    patients = store.list_users(db, role="patient")

    # Per-department breakdown
    dept_load = {}
    for b in bookings_today:
        dept = b.get("department", "Unknown")
        if dept not in dept_load:
            dept_load[dept] = {"total": 0, "checked_in": 0, "in_service": 0, "done": 0, "booked": 0, "no_show": 0}
        dept_load[dept]["total"] += 1
        status = b.get("status", "booked")
        if status in dept_load[dept]:
            dept_load[dept][status] += 1

    # Per-doctor breakdown
    doctor_load = []
    for d in on_duty_today:
        doc_bookings = store.list_bookings(db, doctor_id=d["id"], date=today)
        waiting = sum(1 for b in doc_bookings if b.get("status") in ("checked_in", "in_service"))
        completed = sum(1 for b in doc_bookings if b.get("status") == "done")
        remaining = sum(1 for b in doc_bookings if b.get("status") in ("booked", "checked_in"))
        doctor_load.append({
            "name": d["name"],
            "department": d["department"],
            "shift": f"{d.get('shift_start', '?')}–{d.get('shift_end', '?')}",
            "total_patients": len(doc_bookings),
            "currently_waiting_or_in_service": waiting,
            "completed": completed,
            "remaining_booked": remaining,
        })

    no_show_count = counts.get("no_show", 0)
    total_today = len(bookings_today)
    no_show_rate = f"{(no_show_count / total_today * 100):.1f}%" if total_today > 0 else "0%"

    context = {
        "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "today_date": today,
        "hospital_summary": {
            "total_registered_patients": len(patients),
            "total_doctors_on_roster": len(all_doctors),
            "doctors_on_duty_today": len(on_duty_today),
            "appointments_today": total_today,
            "booked": counts.get("booked", 0),
            "checked_in_waiting": counts.get("checked_in", 0),
            "in_service_now": counts.get("in_service", 0),
            "completed_today": counts.get("done", 0),
            "no_shows_today": no_show_count,
            "no_show_rate_today": no_show_rate,
            "cancelled_today": counts.get("cancelled", 0),
        },
        "department_load": dept_load,
        "doctor_wise_load": doctor_load,
        "doctors_on_duty_names": [d["name"] + " (" + d["department"] + ")" for d in on_duty_today],
    }

    answer = None

    try:
        explainer = service.explainer()

        # Route through explain_clinical_copilot with rich context
        if hasattr(explainer, "explain_clinical_copilot"):
            answer = explainer.explain_clinical_copilot(question, context)
        elif hasattr(explainer, "answer"):
            answer = explainer.answer(question, context)

    except Exception as e:
        answer = None

    if not answer:
        # Fallback: build a readable answer from live data
        answer = _build_fallback_answer(question, context)

    st.session_state["admin_chat"].append({"role": "assistant", "content": answer})
    st.rerun()


def _build_fallback_answer(question: str, ctx: dict) -> str:
    """Build a readable answer from live context when Gemini is unavailable."""
    q = question.lower()
    s = ctx.get("hospital_summary", {})
    dept_load = ctx.get("department_load", {})
    doctor_load = ctx.get("doctor_wise_load", [])
    today = ctx.get("today_date", date.today().isoformat())

    if any(w in q for w in ["waiting", "wait", "queue", "checked in"]):
        waiting = s.get("checked_in_waiting", 0)
        in_service = s.get("in_service_now", 0)
        return (
            f"Currently there are **{waiting}** patient(s) checked in and waiting, "
            f"and **{in_service}** patient(s) in active consultation across all departments. "
            f"Total appointments today: {s.get('appointments_today', 0)}."
        )

    if any(w in q for w in ["doctor", "duty", "on duty", "available"]):
        lines = [f"**{d['name']}** ({d['department']}) — {d['total_patients']} patients, {d['completed']} done, {d['remaining_booked']} remaining, shift {d['shift']}" for d in doctor_load]
        if lines:
            return f"Doctors on duty today ({today}):\n\n" + "\n\n".join(lines)
        return f"No doctors have published duty hours for today ({today}) yet."

    if any(w in q for w in ["department", "dept", "load", "busiest"]):
        if dept_load:
            busiest = max(dept_load.items(), key=lambda x: x[1]["total"])
            lines = [f"**{dept}**: {v['total']} total, {v['checked_in']} waiting, {v['done']} done" for dept, v in dept_load.items()]
            return f"Department load today:\n\n" + "\n\n".join(lines) + f"\n\nBusiest department: **{busiest[0]}** ({busiest[1]['total']} patients)."
        return "No department data available yet for today."

    if any(w in q for w in ["no show", "no-show", "absent", "miss"]):
        return (
            f"Today's no-show count: **{s.get('no_shows_today', 0)}** out of "
            f"{s.get('appointments_today', 0)} appointments "
            f"(no-show rate: {s.get('no_show_rate_today', '0%')})."
        )

    if any(w in q for w in ["summary", "overview", "status", "report"]):
        return (
            f"**Hospital Status — {today}**\n\n"
            f"- Doctors on duty: {s.get('doctors_on_duty_today', 0)}\n"
            f"- Total appointments: {s.get('appointments_today', 0)}\n"
            f"- Checked in & waiting: {s.get('checked_in_waiting', 0)}\n"
            f"- In consultation now: {s.get('in_service_now', 0)}\n"
            f"- Completed: {s.get('completed_today', 0)}\n"
            f"- No-shows: {s.get('no_shows_today', 0)} ({s.get('no_show_rate_today', '0%')})\n"
            f"- Registered patients: {s.get('total_registered_patients', 0)}"
        )

    # Generic fallback
    return (
        f"**Live Hospital Summary ({today})**\n\n"
        f"Appointments today: {s.get('appointments_today', 0)} | "
        f"Waiting: {s.get('checked_in_waiting', 0)} | "
        f"In Service: {s.get('in_service_now', 0)} | "
        f"Completed: {s.get('completed_today', 0)} | "
        f"Doctors on duty: {s.get('doctors_on_duty_today', 0)}\n\n"
        "For more specific details, please ask about doctors, departments, no-shows, or queue status."
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_admin_portal(user):
    """Render the hospital administrator portal."""
    service.require_role(user, ("admin",))

    top1, top2 = st.columns([5, 1])
    with top1:
        st.title("🏥 Hospital Admin Portal")
        st.caption("Live queue operations · Doctors · Patients · AI Copilot")
    with top2:
        if st.button("Logout"):
            _logout()

    st.divider()

    tabs = st.tabs(
        [
            "📊 Dashboard",
            "👨‍⚕️ Doctors",
            "👥 Patients",
            "📅 Appointments",
            "🧠 AI Copilot",
        ]
    )

    with tabs[0]:
        _dashboard(user)

    with tabs[1]:
        _doctor_management()

    with tabs[2]:
        _patient_management()

    with tabs[3]:
        _appointments()

    with tabs[4]:
        _clinical_copilot()
