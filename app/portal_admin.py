"""Hospital administrator portal."""

from __future__ import annotations

from datetime import date

import streamlit as st
import service
import store


def _logout():
    st.session_state["user"] = None
    st.session_state.pop("portal", None)
    st.rerun()


def _add_emergency_patient_section(user):
    st.markdown("### 🚨 Add Emergency Patient to Live Queue")
    st.caption(
        "Admin-authorized clinical triage. Directly inserts an urgent walk-in patient into the department's "
        "live queue ahead of routine waiting patients without requiring an advance appointment slot."
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
            selected_dept = st.selectbox(
                "Department",
                departments,
                key="em_department_select",
            )

        patient_user_id = None
        patient_name = ""

        if patient_type == "Existing Registered Patient":
            patients = store.list_users(db, role="patient")
            if not patients:
                st.warning("No registered patient accounts found. Please choose 'New Walk-In Emergency Patient'.")
            else:
                p_options = {p["id"]: f"{p['full_name']} (@{p['username']}) - Phone: {p.get('phone') or 'N/A'}" for p in patients}
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
                patient_phone = st.text_input("Contact Phone (optional)", placeholder="+1-555-0199", key="em_walkin_phone")

        all_dept_docs = service.list_doctors(department=selected_dept, active_only=True)
        avail_dept_docs = [d for d in all_dept_docs if service.is_doctor_available(d["id"], today_iso)]

        if not avail_dept_docs:
            st.error(f"⚠️ No active doctors are available in {selected_dept} today ({today_iso}). Cannot insert emergency walk-in.")
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
            placeholder="Describe clinical symptoms requiring emergency priority (e.g. Acute chest pain, respiratory distress, trauma)",
            key="em_reason_text",
        )

        if st.button("🚨 Add Emergency Patient to Live Queue", type="primary", use_container_width=True, key="btn_submit_emergency_walkin"):
            if not patient_name or not patient_name.strip():
                st.error("Please provide the patient name.")
                return
            if not doctor_id:
                st.error(f"Cannot assign emergency walk-in: No available doctor in {selected_dept} today.")
                return
            if not emergency_reason or not emergency_reason.strip():
                st.error("Please provide an emergency reason for clinical justification.")
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
                    f"✅ Emergency patient '{patient_name.strip()}' successfully added to {selected_dept} live queue! "
                    f"(Record ID: #{booking_id}, Priority: Emergency). "
                    f"In-app emergency notifications dispatched to waiting patients."
                )
                st.rerun()


def _dashboard(user=None):
    if user is None:
        user = st.session_state.get("user")
    st.subheader("📊 Hospital Dashboard")

    db = service.db_path()

    # Unified Date Selection at the top of the dashboard
    col_date1, col_date2 = st.columns([2, 3])
    with col_date1:
        selected_date = st.date_input(
            "Select Dashboard & Queue Date",
            value=date.today(),
            key="admin_dash_queue_date",
        )
    with col_date2:
        st.write("")
        st.write("")
        view_all_dates = st.checkbox("Show Cumulative All-Time Metrics (All Dates)", value=False, key="admin_dash_view_all")

    target_date_iso = None if view_all_dates else selected_date.isoformat()
    sel_date_formatted = "All Dates" if view_all_dates else selected_date.strftime("%d-%m-%Y")

    all_doctors = service.list_all_hospital_doctors()
    active_doctors = [d for d in all_doctors if d.get("active")]

    # Doctors active and available on the selected date
    if view_all_dates:
        active_and_available = active_doctors
    else:
        active_and_available = [
            d for d in active_doctors
            if service.is_doctor_available(d["id"], selected_date.isoformat())
        ]

    # Real registered patient accounts from SQLite
    patients = store.list_users(
        db,
        role="patient",
    )

    # Real appointments from SQLite
    appointments = store.list_bookings(
        db,
        date=target_date_iso,
    )

    counts = store.counts_by_status(
        db,
        date=target_date_iso,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Active Doctors",
            len(active_and_available),
            help=f"Active & available on {sel_date_formatted} (Total hospital doctors: {len(all_doctors)})",
        )

    with c2:
        st.metric(
            "Registered Patients",
            len(patients),
            help="Total registered patient profiles in SQLite database",
        )

    with c3:
        st.metric(
            f"Appointments ({sel_date_formatted})",
            len(appointments),
            help=f"Bookings recorded for {sel_date_formatted}",
        )

    with c4:
        st.metric(
            "Checked In",
            counts.get("checked_in", 0),
        )

    with c5:
        st.metric(
            "Completed",
            counts.get("done", 0),
        )

    st.divider()

    st.markdown(f"### Appointment Status ({sel_date_formatted})")

    status_order = ["booked", "checked_in", "in_service", "done", "no_show", "cancelled"]
    rows = [
        {
            "Status": (
                "Completed" if k == "done" else k.replace("_", " ").title()
            ),
            "Count": counts.get(k, 0),
        }
        for k in status_order
        if k in counts
    ]
    for k, v in counts.items():
        if k not in status_order:
            rows.append({"Status": k.replace("_", " ").title(), "Count": v})

    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(f"No appointments found for {sel_date_formatted}.")

    st.divider()

    # --- 🚨 Add Emergency Patient to Live Queue Section (Part C) ---
    _add_emergency_patient_section(user)

    st.divider()

    # --- Active & Available Doctors & Patient Queue Section ---
    st.markdown(f"### 👨‍⚕️ Active & Available Doctors — Live Patient Queue ({sel_date_formatted})")
    st.caption("Live doctor-wise queue showing doctors who are active and available on the selected date, with real-time patient statuses from SQLite.")

    col_filter1, col_filter2 = st.columns([2, 2])
    with col_filter1:
        avail_names = ["All Available Doctors"] + [d["name"] for d in active_and_available]
        selected_doc = st.selectbox(
            f"Filter Doctor Queue ({sel_date_formatted})",
            avail_names,
            key="admin_dash_selected_doc",
        )

    sel_date_iso = selected_date.isoformat()

    doctors_to_show = active_and_available
    if selected_doc != "All Available Doctors":
        doctors_to_show = [d for d in active_and_available if d["name"] == selected_doc]

    if not active_and_available:
        st.warning(
            f"No doctors have published duty hours on {sel_date_formatted}. "
            f"Doctors open slots from the Doctor portal (Shift & Timetable)."
        )
    elif not doctors_to_show:
        st.info("Selected doctor is not available on this date.")
    else:
        lifecycle_display = {
            "booked": "Booked",
            "checked_in": "Checked In",
            "in_service": "In Service",
            "done": "Completed",
            "no_show": "No Show",
            "cancelled": "Cancelled",
        }

        for doc in doctors_to_show:
            doc_appts = store.list_bookings(
                db,
                doctor_id=doc["id"],
                date=sel_date_iso,
            )
            patient_count = len(doc_appts)
            patient_label = f"{patient_count} {'Patient' if patient_count == 1 else 'Patients'}"

            expander_title = f"👨‍⚕️ {doc['name']} — {doc['department']} — {patient_label}"
            with st.expander(expander_title, expanded=(patient_count > 0)):
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.markdown(f"**Doctor:** {doc['name']}")
                    st.markdown(f"**Department:** {doc['department']}")
                with dc2:
                    st.markdown(f"**Shift:** {doc['shift_start']} - {doc['shift_end']}")
                    st.markdown("**Status:** 🟢 Active & Available")
                with dc3:
                    st.markdown(f"**Patients Assigned:** {patient_count}")

                st.markdown("#### Patient Details & Queue Status")
                if patient_count == 0:
                    st.info("0 patients currently assigned or scheduled for this doctor.")
                else:
                    for idx, appt in enumerate(doc_appts, start=1):
                        p_name = appt.get("patient_name", "Unknown Patient")
                        raw_status = appt.get("status", "booked")
                        status_str = lifecycle_display.get(
                            raw_status,
                            raw_status.replace("_", " ").title(),
                        )
                        slot_str = f" · Slot: {appt['slot_start']}" if appt.get("slot_start") else ""
                        prio_str = " · 🚨 **Emergency**" if appt.get("priority_level", 0) >= 1 else ""
                        st.markdown(
                            f"**{idx}. {p_name}** — `{status_str}`{slot_str}{prio_str}"
                        )


def _doctor_management():
    st.subheader("👨‍⚕️ Doctor & Hospital Roster Management")

    db = service.db_path()

    # Retrieve ALL hospital doctors (active AND inactive) with login credentials
    doctors = service.list_all_hospital_doctors()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Hospital Doctors", len(doctors))
    with m2:
        st.metric("Active Doctors", sum(1 for d in doctors if d.get("active")))
    with m3:
        st.metric("Inactive / Off Duty", sum(1 for d in doctors if not d.get("active")))

    # -------------------------------------------------------------
    # 1. ALL HOSPITAL DOCTORS (DOCTOR INFORMATION ONLY)
    # -------------------------------------------------------------
    st.markdown("### 📋 All Hospital Doctors")
    st.caption("Complete hospital doctor roster information. Login credentials are listed separately below.")

    if doctors:
        doctor_info_table = [
            {
                "Doctor ID": d["id"],
                "Doctor Name": d["name"],
                "Department": d["department"],
                "Status": "Active" if d.get("active") else "Inactive",
                "Default Shift Start": d.get("shift_start", "09:00"),
                "Default Shift End": d.get("shift_end", "13:00"),
            }
            for d in doctors
        ]

        st.dataframe(
            doctor_info_table,
            hide_index=True,
            use_container_width=True,
        )

    st.info(
        "Duty hours and bookable slots are published by each doctor when they "
        "set their shift for a day. Admin does not mark date availability — "
        "ML packs slots inside the doctor's published window, and early finishes "
        "automatically open new slots from remaining duty time."
    )

    # -------------------------------------------------------------
    # 2. ON-DUTY TODAY (read-only, doctor-published)
    # -------------------------------------------------------------
    st.divider()
    st.markdown("### 🟢 Doctors On Duty (published by doctors)")
    st.caption(
        "Read-only view of duty days doctors have published. "
        "Buffer between slots is dynamic (crowd / no-show / overrun aware), not admin-fixed."
    )
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

    # -------------------------------------------------------------
    # 3. DOCTOR LOGIN CREDENTIALS
    # -------------------------------------------------------------
    st.divider()
    st.markdown("### 🔐 Doctor Login Credentials")
    st.caption("Secure administrative view of login credentials linked to hospital doctor accounts.")

    if doctors:
        credentials_table = [
            {
                "Doctor": d["name"],
                "Department": d["department"],
                "Username": d.get("username") or "—",
                "Password": d.get("login_password") or "doctor123",
            }
            for d in doctors
        ]

        st.dataframe(
            credentials_table,
            hide_index=True,
            use_container_width=True,
        )

    # -------------------------------------------------------------
    # 4. UPDATE DOCTOR STATUS
    # -------------------------------------------------------------
    st.divider()
    st.markdown("### ⚙️ Update Doctor Status")
    st.caption("Change a doctor's active or inactive status in the hospital roster.")

    if doctors:
        col_u1, col_u2, col_u3 = st.columns([3, 2, 2])
        with col_u1:
            selected_id = st.selectbox(
                "Select Doctor to Update",
                options=[d["id"] for d in doctors],
                format_func=lambda did: next(
                    (f"{d['name']} ({d['department']}) — {'Active' if d.get('active') else 'Inactive'}" for d in doctors if d["id"] == did),
                    str(did),
                ),
                key="admin_manage_doc_select",
            )
        target_doc = next(d for d in doctors if d["id"] == selected_id)
        with col_u2:
            target_status = st.selectbox(
                "Set Status",
                options=[1, 0],
                index=0 if target_doc.get("active") else 1,
                format_func=lambda x: "Active" if x == 1 else "Inactive",
                key="admin_manage_doc_status",
            )
        with col_u3:
            st.write("")
            st.write("")
            if st.button("Update Status", type="primary", key="btn_update_doc_status"):
                store.set_doctor_status(db, selected_id, target_status)
                status_text = "Active" if target_status == 1 else "Inactive"
                st.success(f"{target_doc['name']} is now set to {status_text}.")
                st.rerun()

    # -------------------------------------------------------------
    # 5. ADD DOCTOR
    # -------------------------------------------------------------
    st.divider()
    st.markdown("### ➕ Add Doctor")
    st.caption("Register a new clinical doctor roster record and create their linked login account.")

    departments = service.specialty_departments()

    name = st.text_input(
        "Doctor name",
        placeholder="e.g. Dr. Ramesh Gupta",
        key="admin_doc_name",
    )

    department = st.selectbox(
        "Department",
        departments,
        key="admin_doc_department",
    )

    c1, c2 = st.columns(2)
    with c1:
        shift_start = st.text_input(
            "Shift start",
            value="09:00",
            key="admin_doc_shift_start",
        )
    with c2:
        shift_end = st.text_input(
            "Shift end",
            value="13:00",
            key="admin_doc_shift_end",
        )

    default_uname = ""
    if name.strip():
        clean_n = name.strip().lower().replace("dr.", "").replace("dr", "").strip()
        default_uname = "dr_" + "_".join(clean_n.split())

    c3, c4 = st.columns(2)
    with c3:
        username = st.text_input(
            "Doctor Username",
            value=default_uname,
            placeholder="e.g. dr_gupta",
            key="admin_doc_username",
        )
    with c4:
        password = st.text_input(
            "Doctor Login Password",
            value="doctor123",
            key="admin_doc_password",
        )

    active = st.checkbox(
        "Active doctor",
        value=True,
        key="admin_doc_active",
    )

    if st.button(
        "Add Doctor",
        type="primary",
        key="btn_add_doctor",
    ):
        if not name.strip():
            st.error("Enter the doctor's name.")
            return

        if not username.strip():
            st.error("Enter a doctor username.")
            return

        if not password or len(password) < 4:
            st.error("Password must be at least 4 characters long.")
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
                f"Doctor created successfully! ID: {doctor_id}, Username: {username.strip()}. "
                f"They must publish duty hours from the Doctor portal to open bookable slots."
            )
            st.rerun()

        except Exception as exc:
            st.error(f"Could not add doctor: {exc}")


def _patient_management():
    st.subheader("👥 Patient Accounts")

    patients = store.list_users(
        service.db_path(),
        role="patient",
    )

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
                "Created": p.get("created_at"),
            }
            for p in patients
        ],
        hide_index=True,
        use_container_width=True,
    )


def _appointments():
    st.subheader("📅 Appointment Management")

    db = service.db_path()

    selected_date = st.date_input(
        "Appointment date",
        value=date.today(),
        key="admin_appt_date",
    )

    bookings = store.list_bookings(
        db,
        date=selected_date.isoformat(),
    )

    if not bookings:
        st.info(
            "No appointments found for the selected date."
        )
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
                "Priority": (
                    "EMERGENCY"
                    if b.get("priority_level", 0) >= 1
                    else "Routine"
                ),
                "Status": b["status"],
                "No-show probability": (
                    f"{float(b['noshow_prob']) * 100:.1f}%"
                    if b.get("noshow_prob") is not None
                    else "—"
                ),
            }
        )

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
    )


def _analytics():
    st.subheader("📈 Analytics & Model Calibration")

    metrics = service.model_metrics()
    buffer_report = service.buffer_report()
    policy = service.policy_comparison()

    if metrics:
        st.markdown("### 🤖 Base Model Metrics & Feature Importances")
        st.json(metrics)
    else:
        st.info("model_metrics.json is not available yet.")

    # --- Novelty 1: Conformal Prediction Bounds ---
    conf = service.conformal_metrics()
    if conf:
        st.markdown("### 🎯 Conformal Prediction (Guaranteed 90% Uncertainty Intervals)")
        st.caption("Distribution-free, finite-sample calibrated prediction intervals for waiting and service times.")
        st.json(conf)

    # --- Novelty 2: Pareto Frontier Efficiency ---
    pareto = service.pareto_report()
    if pareto:
        st.markdown("### ⚖️ Multi-Objective Pareto Frontier Efficiency")
        st.caption("Empirical non-dominated Pareto frontier benchmark vs. real-time greedy dispatching policy.")
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.metric("Pareto Efficiency Ratio", pareto.get("efficiency_percentage", "100.0%"))
        with col_p2:
            st.write(pareto.get("significance", ""))
        st.json(pareto.get("sample_cohort", {}))

    # --- Novelty 3: Chance-Constrained Overbooking Buffer ---
    chance_buf = service.chance_constrained_report()
    if chance_buf:
        st.markdown("### 🛡️ Chance-Constrained Overbooking Buffer")
        st.caption("Poisson-Binomial tail bounds ensuring P(overcrowding) <= 5% risk tolerance.")
        st.json(chance_buf)

    # --- Novelty 4: Algorithmic Fairness Audit ---
    fair = service.fairness_report()
    if fair:
        st.markdown("### 🔍 Algorithmic Fairness & Group Parity Audit")
        st.caption("Audits Disparate Impact Ratio (80% rule), Demographic Parity, and Waiting Time Equity.")
        dept_fair = fair.get("classification_fairness_department", {})
        f1, f2, f3 = st.columns(3)
        with f1:
            st.metric("Disparate Impact Ratio", dept_fair.get("disparate_impact_ratio", "N/A"))
        with f2:
            st.metric("Demographic Parity Diff", dept_fair.get("demographic_parity_difference", "N/A"))
        with f3:
            st.metric("80% Rule Satisfied", "✅ PASS" if dept_fair.get("four_fifths_rule_satisfied") else "❌ FAIL")
        st.json(fair)

    # --- Novelty 5: Counterfactual What-If Scenarios ---
    cf_data = service.counterfactual_report()
    if cf_data:
        st.markdown("### 🔮 Counterfactual 'What-If' Explanations")
        st.caption("Causal counterfactual deltas across punctuality, emergency preemption, and physician staffing.")
        st.json(cf_data)

    if buffer_report:
        st.markdown("### 📊 Dynamic Buffer Analysis")
        st.json(buffer_report)

    if policy:
        st.markdown("### ⚙️ Historical Policy Evaluation")
        st.json(policy)

    st.divider()

    st.markdown("### Evaluation Figures")

    figure_names = [
        "no_show_calibration.png",
        "no_show_feature_importance.png",
        "service_time_pred_vs_actual.png",
        "service_time_feature_importance.png",
        "buffer_analysis.png",
    ]

    for name in figure_names:
        path = service.figure_path(name)

        if path:
            st.markdown(f"#### {name}")
            st.image(str(path), use_container_width=True)


def _policy_benchmark():
    st.subheader("⚙️ Multi-Objective Policy Benchmark")

    st.caption(
        "Adjust the objective weights and inspect the available "
        "policy evaluation artifact."
    )

    c1, c2 = st.columns(2)

    with c1:
        waiting = st.slider(
            "Waiting-time weight",
            0.0,
            1.0,
            0.4,
            0.05,
        )

        utilization = st.slider(
            "Doctor-utilization weight",
            0.0,
            1.0,
            0.2,
            0.05,
        )

    with c2:
        fairness = st.slider(
            "Fairness weight",
            0.0,
            1.0,
            0.2,
            0.05,
        )

        emergency = st.slider(
            "Emergency-responsiveness weight",
            0.0,
            1.0,
            0.2,
            0.05,
        )

    total = (
        waiting
        + utilization
        + fairness
        + emergency
    )

    st.caption(
        f"Weight total: {total:.2f}"
    )

    if st.button(
        "Evaluate Policies",
        type="primary",
    ):
        if total <= 0:
            st.error(
                "At least one objective weight must be greater than zero."
            )
            return

        weights = {
            "waiting": waiting / total,
            "utilization": utilization / total,
            "fairness": fairness / total,
            "emergency": emergency / total,
        }

        try:
            result = service.evaluate_policies_live(
                weights=weights,
                emergency_level=1,
            )

            st.session_state[
                "policy_result"
            ] = result

        except Exception as exc:
            st.error(
                f"Policy evaluation failed: {exc}"
            )

    result = st.session_state.get(
        "policy_result"
    )

    if not result:
        return

    st.markdown("### Evaluation Result")

    st.json(result)

    policies = result.get("policies", {})

    if policies:
        rows = []

        for name, values in policies.items():
            row = {
                "Policy": name,
            }

            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(
                        value,
                        (str, int, float, bool),
                    ):
                        row[key] = value

            rows.append(row)

        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
        )


def _clinical_copilot():
    st.subheader("🧠 Clinical / Administrative AI Copilot")

    if "admin_chat" not in st.session_state:
        st.session_state["admin_chat"] = []

    for message in st.session_state["admin_chat"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask about clinic operations..."
    )

    if not question:
        return

    st.session_state["admin_chat"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    db = service.db_path()
    today = date.today().isoformat()

    bookings = store.list_bookings(
        db,
        date=today,
    )

    doctors = store.list_doctors(
        db,
        active_only=True,
    )

    context = {
        "date": today,
        "active_doctors": len(doctors),
        "appointments_today": len(bookings),
        "status_counts": store.counts_by_status(
            db,
            date=today,
        ),
    }

    answer = None

    try:
        explainer = service.explainer()

        if hasattr(
            explainer,
            "explain_clinical_copilot",
        ):
            answer = explainer.explain_clinical_copilot(
                question,
                context,
            )
        elif hasattr(explainer, "answer"):
            answer = explainer.answer(
                question,
                context,
            )

    except Exception:
        answer = None

    if not answer:
        answer = (
            "The AI copilot is currently unavailable. "
            "Use the dashboard and analytics tabs for the "
            "current clinic information."
        )

    st.session_state["admin_chat"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.rerun()


def render_admin_portal(user):
    """Render the hospital administrator portal."""

    service.require_role(
        user,
        ("admin",),
    )

    top1, top2 = st.columns([5, 1])

    with top1:
        st.title("🏥 Hospital Admin Portal")
        st.caption(
            "Queue operations · Doctors · Patients · "
            "Analytics · Optimization"
        )

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
            "📈 Analytics",
            "⚙️ Policy Benchmark",
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
        _analytics()

    with tabs[5]:
        _policy_benchmark()

    with tabs[6]:
        _clinical_copilot()