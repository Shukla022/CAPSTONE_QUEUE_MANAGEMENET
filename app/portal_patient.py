"""Patient portal for the Queue Management System."""

from __future__ import annotations

from datetime import date, timedelta
import math

import streamlit as st
import scheduling
import service
import store


def _logout():
    st.session_state["user"] = None
    st.session_state.pop("portal", None)
    st.rerun()


def _safe_value(value, default="—"):
    if value is None or value == "":
        return default
    return value


def _format_probability(value):
    if value is None:
        return "Unavailable"

    try:
        value = float(value)

        if math.isnan(value):
            return "Unavailable"

        return f"{value * 100:.1f}%"
    except Exception:
        return "Unavailable"


def _status_stepper(status):
    statuses = [
        ("booked", "📅 Booked"),
        ("checked_in", "✅ Checked In"),
        ("in_service", "🩺 In Consultation"),
        ("done", "✔️ Completed"),
    ]

    order = {
        "booked": 0,
        "checked_in": 1,
        "in_service": 2,
        "done": 3,
    }

    current = order.get(status, 0)

    cols = st.columns(4)

    for i, (key, label) in enumerate(statuses):
        with cols[i]:
            if i < current:
                st.success(label)
            elif i == current:
                st.info(label)
            else:
                st.caption(label)


def _live_ticket(user):
    st.header("🎫 Live Queue Ticket")

    user_bookings = service.list_patient_bookings(patient_user_id=user["id"])
    if not user_bookings:
        st.info("You do not currently have any booked appointments. Visit the **📅 Book Appointment** tab to schedule a consultation.")
        return

    status_emoji = {
        "booked": "📅 Booked",
        "checked_in": "✅ Checked In",
        "in_service": "🩺 In Consultation",
        "done": "✔️ Completed",
        "no_show": "⚠️ No Show",
        "cancelled": "❌ Cancelled",
    }

    # Find default selection
    default_ticket = service.get_patient_live_ticket(patient_user_id=user["id"])
    default_id = default_ticket["id"] if default_ticket else user_bookings[0]["id"]

    current_sel_id = st.session_state.get("selected_patient_booking_id", default_id)
    if not any(b["id"] == current_sel_id for b in user_bookings):
        current_sel_id = default_id

    booking_options = []
    id_map = {}
    default_index = 0
    for idx, b in enumerate(user_bookings):
        b_id = b["id"]
        b_date = scheduling.format_date_human(b.get("appointment_date"))
        b_time = scheduling.format_clock_12h(b.get("slot_start"))
        b_dept = b.get("department", "Clinic")
        b_st = status_emoji.get(b.get("status"), b.get("status", ""))
        label = f"#{b_id}: {b_date} at {b_time} — {b_dept} [{b_st}]"
        booking_options.append(label)
        id_map[label] = b_id
        if b_id == current_sel_id:
            default_index = idx

    st.markdown("### 📑 My Appointments")
    selected_label = st.selectbox(
        "Choose an appointment to view its live queue ticket and status:",
        booking_options,
        index=default_index,
        key="patient_appointment_selector",
    )
    selected_booking_id = id_map[selected_label]
    st.session_state["selected_patient_booking_id"] = selected_booking_id

    ticket = service.get_patient_live_ticket(
        patient_user_id=user["id"],
        booking_id=selected_booking_id,
    )

    if not ticket:
        st.info("Could not load details for this appointment.")
        return

    status = ticket.get("status", "booked")
    appt_date = ticket.get("appointment_date", "")
    is_past_appt = bool(appt_date) and appt_date < date.today().isoformat()

    # Only show the progress stepper for the four main lifecycle statuses.
    if status in ("booked", "checked_in", "in_service", "done"):
        _status_stepper(status)

    st.divider()

    # Dynamic metrics display matching the appointment status lifecycle
    if status == "booked":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Booked")
        with col2:
            st.metric("Queue Position", "Available after check-in")
        with col3:
            st.metric("Estimated Turn", ticket.get("estimated_turn") or ticket.get("appointment_time_12h") or "—")
        with col4:
            st.metric("Department", _safe_value(ticket.get("doctor_department")))

        st.info("📌 You have a scheduled appointment. You are not currently waiting in the live queue. Please check in when you arrive at the clinic.")

    elif status == "checked_in":
        # In-app emergency notifications for waiting patients (Part C4)
        active_notifs = ticket.get("notifications") or service.get_waiting_patient_notifications(
            ticket.get("doctor_department", ""), ticket.get("doctor_id")
        )
        dismissed = st.session_state.get("dismissed_alerts", set())
        unack = [n for n in active_notifs if n.get("id") not in dismissed]
        for notif in unack:
            col_msg, col_btn = st.columns([5, 1])
            with col_msg:
                st.warning(f"🚨 {notif['message']}")
            with col_btn:
                if st.button("Acknowledge", key=f"ack_alert_{notif['id']}"):
                    if "dismissed_alerts" not in st.session_state:
                        st.session_state["dismissed_alerts"] = set()
                    st.session_state["dismissed_alerts"].add(notif["id"])
                    st.rerun()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            q_pos = ticket.get("queue_position", "#1")
            q_display = str(q_pos) if str(q_pos).startswith("#") else f"#{q_pos}"
            st.metric("Queue Position", q_display)
        with col2:
            ahead_val = ticket.get("patients_ahead", 0)
            st.metric("Patients Ahead", ahead_val if ahead_val is not None else 0)
        with col3:
            st.metric("Estimated Wait", ticket.get("live_eta_display", "0 minutes"))
        with col4:
            st.metric("Estimated Turn", ticket.get("estimated_turn", "—"))

        if "Shift Concluded" in str(ticket.get("live_eta_display", "")):
            st.warning("⚠️ The doctor's shift for this appointment has concluded. Please contact reception or rebook.")
        else:
            st.success("✅ You are checked in. Please remain available in the waiting area for your consultation.")

    elif status == "in_service":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "In Consultation")
        with col2:
            st.metric("Queue Position", "Current Patient")
        with col3:
            st.metric("Estimated Turn", "Now")
        with col4:
            st.metric("Department", _safe_value(ticket.get("doctor_department")))

        st.info("🩺 Your consultation is currently in progress.")

    elif status == "done":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Completed")
        with col2:
            st.metric("Queue Position", "Completed")
        with col3:
            st.metric("Consultation", "Completed")
        with col4:
            st.metric("Department", _safe_value(ticket.get("doctor_department")))

        st.success("✔️ This consultation has been completed. Full records are available in Medical History.")

    elif status in ("no_show", "cancelled"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            label = "No Show" if status == "no_show" else "Cancelled"
            st.metric("Status", label)
        with col2:
            st.metric("Appointment Date", ticket.get("appointment_date_human") or "—")
        with col3:
            st.metric("Doctor", ticket.get("doctor_name") or "—")
        with col4:
            st.metric("Department", _safe_value(ticket.get("doctor_department")))
        if status == "no_show":
            st.warning("⚠️ This appointment was not attended and has been marked as no-show. Please book a new appointment.")
        else:
            st.warning("⚠️ This appointment was cancelled. Please book a new appointment if needed.")

    st.markdown("### Appointment Details")

    details = {
        "Status": status.replace("_", " ").title() if status != "in_service" else "In Consultation",
        "Appointment Date": ticket.get("appointment_date_human") or ticket.get("appointment_date"),
        "Appointment Time": ticket.get("appointment_time_12h") or ticket.get("slot_start"),
        "Doctor": ticket.get("doctor_name"),
        "Department": ticket.get("doctor_department"),
        "Doctor Shift": ticket.get("doctor_shift"),
        "Priority": (
            "🚨 Emergency"
            if ticket.get("priority_level", 0) >= 1
            else "Routine"
        ),
    }

    st.dataframe(
        [
            {"Field": key, "Value": _safe_value(value)}
            for key, value in details.items()
        ],
        hide_index=True,
        width="stretch",
    )

    if status == "booked":
        if is_past_appt:
            st.warning(
                "⚠️ This appointment has expired. Check-in is no longer available. "
                "Please book a new appointment."
            )
        else:
            is_shift_ended = "Shift Concluded" in str(ticket.get("estimated_turn", ""))
            if is_shift_ended:
                st.warning(
                    f"⚠️ The doctor's shift for this appointment has concluded ({ticket.get('doctor_shift')}). "
                    "Check-in is closed for today."
                )
            else:
                if st.button(
                    "✅ Check in at clinic",
                    type="primary",
                    width="stretch",
                    key=f"check_in_btn_{ticket['id']}",
                ):
                    try:
                        service.patient_check_in(user, ticket["id"])
                        st.success("You are checked in and have entered the live queue!")
                        st.session_state["selected_patient_booking_id"] = ticket["id"]
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Check-in failed: {exc}")

    st.divider()

    st.markdown("### 🤖 Queue Guidance")

    if status == "booked":
        if is_past_appt:
            st.warning(
                "⚠️ This appointment has expired. Check-in is no longer available. "
                "Please book a new appointment."
            )
        else:
            appt_d = ticket.get("appointment_date_human") or ticket.get("appointment_date", "your scheduled date")
            appt_t = ticket.get("appointment_time_12h") or ticket.get("slot_start", "")
            doc_n = ticket.get("doctor_name", "the doctor")
            dept_n = ticket.get("doctor_department", "the clinic")
            st.info(
                f"📌 Your appointment is confirmed for **{appt_d}** at **{appt_t}** with **{doc_n}** ({dept_n}). "
                "You are not currently waiting in the live queue. Live queue position and waiting time estimates "
                "will become active when you check in at the clinic on the day of your appointment."
            )
    elif status == "done":
        st.success("✔️ Your consultation has concluded. Your medical history and doctor's notes have been recorded.")
    elif status in ("no_show", "cancelled"):
        if status == "no_show":
            st.warning("⚠️ This appointment was not attended. Please book a new appointment if you still need care.")
        else:
            st.info("📌 This appointment was cancelled. Please book a new appointment if needed.")
    else:
        # status in ("checked_in", "in_service") -> active live queue
        explainer = service.explainer()
        try:
            context = {
                "queue_position": ticket.get("queue_position"),
                "live_eta_minutes": ticket.get("live_eta_minutes"),
                "predicted_wait": ticket.get("live_eta_minutes") or 0.0,
                "queue_length": ticket.get("patients_ahead", 0),
                "status": status,
                "department": ticket.get("doctor_department"),
                "doctor": ticket.get("doctor_name"),
            }

            if hasattr(explainer, "explain_wait_estimate"):
                response = explainer.explain_wait_estimate(context)
                st.info(response)
            else:
                st.info(
                    "Your live ETA is calculated from the doctor's current "
                    "queue timeline. Please remain available in the waiting area."
                )
        except Exception:
            st.info(
                "AI guidance is currently unavailable. "
                "Your queue position and ETA above remain available."
            )


def _booking_tab(user):
    st.header("📅 AI-Powered Appointment Booking")

    if "patient_booking_success_msg" in st.session_state:
        st.success(st.session_state.pop("patient_booking_success_msg"))

    form_ver = st.session_state.get("patient_booking_form_version", 0)

    departments = service.specialty_departments()

    department = st.selectbox(
        "Department",
        departments,
        key=f"patient_booking_department_v{form_ver}",
    )

    appointment_date = st.date_input(
        "Appointment date",
        min_value=date.today(),
        value=date.today() + timedelta(days=1),
        key=f"patient_booking_date_v{form_ver}",
    )
    date_iso = appointment_date.isoformat()

    # Only doctors who published duty hours for this date are bookable
    doctors = service.list_available_doctors_for_date(
        department=department,
        date_str=date_iso,
    )

    if not doctors:
        st.warning(
            "No doctor has published duty hours for this department/date yet. "
            "Slots open after the doctor sets their shift for the day."
        )
        return

    doctor_names = [
        f"{d['name']} — {d.get('shift_start', '09:00')} to {d.get('shift_end', '13:00')}"
        for d in doctors
    ]

    selected_index = st.selectbox(
        "Doctor (on duty)",
        range(len(doctors)),
        format_func=lambda i: doctor_names[i],
        key=f"patient_booking_doctor_v{form_ver}",
    )

    doctor = doctors[selected_index]
    doctor = service.doctor_with_duty(doctor, date_iso)

    is_avail, avail_reason = service.is_doctor_available_on_date(doctor["id"], date_iso)
    if not is_avail:
        st.warning(avail_reason)
        return

    st.markdown("### Available Slots")
    st.caption(
        "Slots are ML-packed from predicted service time with a crowd-aware dynamic buffer. "
        "Early finishes automatically open extra slots in remaining duty time."
    )

    try:
        slots = service.available_slots(
            doctor,
            date_iso,
        )
    except Exception as exc:
        st.error(f"Unable to load available slots: {exc}")
        slots = []

    if not slots:
        st.warning(
            "No ML-generated slots are currently available for this doctor/date."
        )

        st.caption(
            "This can happen when the service-time model is not trained "
            "or all calculated capacity is occupied."
        )
        return

    def slot_label(slot):
        start = slot.get("slot_start") or slot.get("start")
        end = slot.get("slot_end") or slot.get("end")

        if end:
            return f"{start} – {end}"

        return str(start)

    selected_slot = st.selectbox(
        "Choose a slot",
        slots,
        format_func=slot_label,
        key=f"patient_booking_slot_v{form_ver}",
    )

    start_time = (
        selected_slot.get("slot_start")
        or selected_slot.get("start")
    )

    st.divider()

    # ML preview
    hour = 9

    try:
        hour = int(str(start_time).split(":")[0])
    except Exception:
        pass

    day_name = appointment_date.strftime("%A")

    prediction_inputs = {
        "department": department,
        "day_of_week": day_name,
        "time_slot": (
            "Morning"
            if hour < 12
            else "Afternoon"
            if hour < 17
            else "Evening"
        ),
        "hour_of_day": hour,
        "priority_level": 0,
        "active_doctors_count": len(doctors),
        "slot_arrival_count": 1,
        "lead_time_hours": max(
            0,
            (
                appointment_date - date.today()
            ).days * 24,
        ),
        "previous_no_show_count": 0,
    }

    try:
        service_minutes = service.predict_service(
            prediction_inputs
        )
    except Exception:
        service_minutes = float("nan")

    try:
        no_show_prob, drivers = service.predict_no_show(
            prediction_inputs
        )
    except Exception:
        no_show_prob = float("nan")
        drivers = []

    try:
        arrival_delay = service.predict_arrival_delay(
            prediction_inputs
        )
    except Exception:
        arrival_delay = float("nan")

    c1, c2, c3 = st.columns(3)

    with c1:
        if service_minutes == service_minutes:
            st.metric(
                "Predicted Consultation",
                f"{service_minutes:.1f} min",
            )
        else:
            st.metric("Predicted Consultation", "Unavailable")

    with c2:
        st.metric(
            "No-Show Probability",
            _format_probability(no_show_prob),
        )

    with c3:
        if arrival_delay == arrival_delay:
            st.metric(
                "Arrival Forecast",
                f"{arrival_delay:+.1f} min",
            )
        else:
            st.metric("Arrival Forecast", "Unavailable")

    if drivers:
        st.caption(
            "Prediction signals: " + "; ".join(drivers)
        )

    if st.button(
        "Book Appointment",
        type="primary",
        use_container_width=True,
    ):
        is_avail, avail_reason = service.is_doctor_available_on_date(doctor["id"], date_iso)
        if not is_avail:
            st.error(f"Cannot book appointment: {avail_reason}")
            return

        try:
            booking = {
                "patient_name": user["full_name"],
                "patient_user_id": user["id"],
                "department": department,
                "doctor_id": doctor["id"],
                "appointment_date": date_iso,
                "time_slot": (
                    "Morning"
                    if hour < 12
                    else "Afternoon"
                    if hour < 17
                    else "Evening"
                ),
                "slot_start": start_time,
                "slot_end": (
                    selected_slot.get("slot_end")
                    or selected_slot.get("end")
                ),
                "day_of_week": day_name,
                "priority_level": 0,
                "lead_time_hours": prediction_inputs[
                    "lead_time_hours"
                ],
                "previous_no_show_count": 0,
                "noshow_prob": (
                    float(no_show_prob)
                    if no_show_prob == no_show_prob
                    else None
                ),
                "predicted_arrival_delay": (
                    float(arrival_delay)
                    if arrival_delay == arrival_delay
                    else None
                ),
                "status": "booked",
                "emergency_reason": "",
                "assigned_by": "",
            }

            booking_id, err = service.book_appointment(booking)
            if err:
                st.error(err)
                return

            st.session_state["patient_booking_success_msg"] = (
                f"Appointment booked successfully! Booking ID: #{booking_id}"
            )
            st.session_state["selected_patient_booking_id"] = booking_id
            # Reset form for subsequent bookings
            st.session_state["patient_booking_form_version"] = form_ver + 1

            st.rerun()

        except Exception as exc:
            st.error(f"Could not create appointment: {exc}")


def _history_tab(user):
    st.header("📋 Medical History & Past Records")

    try:
        history = service.get_patient_medical_history(
            user,
            user["id"],
        )
    except PermissionError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unable to load medical history: {exc}")
        return

    if not history:
        st.info("No medical records are available yet.")
        return

    for record in history:
        with st.expander(
            f"{record.get('appointment_date', 'Unknown date')} — "
            f"{record.get('department', 'Unknown department')}"
        ):
            c1, c2 = st.columns(2)

            with c1:
                st.write(
                    "**Doctor:**",
                    _safe_value(record.get("doctor_name")),
                )
                st.write(
                    "**Department:**",
                    _safe_value(record.get("doctor_department")),
                )

            with c2:
                st.write(
                    "**Status:**",
                    _safe_value(record.get("status")),
                )
                st.write(
                    "**Service Duration:**",
                    (
                        f"{record.get('service_minutes')} min"
                        if record.get("service_minutes") is not None
                        else "—"
                    ),
                )

            if record.get("notes"):
                st.markdown("**Clinical Notes**")
                st.write(record["notes"])

            if record.get("emergency_reason"):
                st.warning(
                    f"Emergency reason: {record['emergency_reason']}"
                )


def _copilot_tab(user):
    st.header("🤖 Patient AI / Queue Copilot")

    st.caption(
        "Ask about your appointment, queue status, arrival timing, "
        "or general clinic procedures."
    )

    ticket = service.get_patient_live_ticket(
        patient_user_id=user["id"]
    )

    sanitized_appointment = None
    if ticket:
        sanitized_appointment = {
            "status": ticket.get("status"),
            "appointment_date": ticket.get("appointment_date_human") or ticket.get("appointment_date"),
            "appointment_time": ticket.get("appointment_time_12h") or ticket.get("slot_start"),
            "doctor_name": ticket.get("doctor_name"),
            "department": ticket.get("doctor_department"),
            "queue_position": ticket.get("queue_position"),
            "patients_ahead": ticket.get("patients_ahead", 0),
            "estimated_wait": ticket.get("live_eta_display") or ticket.get("live_eta_minutes"),
            "estimated_turn": ticket.get("estimated_turn"),
        }

    context = {
        "patient_name": user.get("full_name", user.get("username")),
        "appointment": sanitized_appointment,
    }

    if "patient_chat" not in st.session_state:
        st.session_state["patient_chat"] = []

    for message in st.session_state["patient_chat"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask your Queue Copilot..."
    )

    if not question:
        return

    st.session_state["patient_chat"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    answer = None

    try:
        explainer = service.explainer()

        if hasattr(explainer, "explain_patient_copilot"):
            answer = explainer.explain_patient_copilot(
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
            "I can help with your queue and appointment information. "
            "For medical diagnosis or treatment decisions, please consult "
            "a qualified healthcare professional."
        )

    st.session_state["patient_chat"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.rerun()


def render_patient_portal(user):
    """Render the complete patient portal."""

    service.require_role(user, ("patient",))
    service.auto_handle_expired_appointments()

    top1, top2 = st.columns([5, 1])

    with top1:
        st.title("👤 Patient Portal")
        st.caption(
            f"Welcome, {user.get('full_name', user.get('username'))}"
        )

    with top2:
        if st.button("Logout"):
            _logout()

    st.divider()

    tabs = st.tabs(
        [
            "🎫 Live Queue",
            "📅 Book Appointment",
            "📋 Medical History",
            "🤖 AI Copilot",
        ]
    )

    with tabs[0]:
        _live_ticket(user)

    with tabs[1]:
        _booking_tab(user)

    with tabs[2]:
        _history_tab(user)

    with tabs[3]:
        _copilot_tab(user)