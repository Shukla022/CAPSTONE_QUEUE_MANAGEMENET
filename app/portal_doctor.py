"""Doctor clinical queue console."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st
import scheduling
import service
import store


def _logout():
    st.session_state["user"] = None
    st.session_state.pop("portal", None)
    st.rerun()


def _status_badge(status):
    mapping = {
        "booked": "📅 Booked",
        "checked_in": "✅ Checked In",
        "in_service": "🩺 In Service",
        "done": "✔️ Completed",
        "no_show": "❌ No Show",
        "cancelled": "🚫 Cancelled",
    }

    return mapping.get(
        status,
        str(status).replace("_", " ").title(),
    )


def _live_queue(user, doctor):
    st.subheader("📋 Live Queue Console")

    today = date.today().isoformat()

    if not service.is_doctor_available(doctor["id"], today):
        st.warning(
            f"You have not published duty hours for today ({scheduling.format_date_human(today)}). "
            f"Open the **Shift & Timetable** tab to publish your shift and open ML-packed slots."
        )

    try:
        schedule = service.live_schedule(
            doctor,
            today,
        )
    except Exception as exc:
        st.error(f"Unable to load live schedule: {exc}")
        return

    timeline = schedule.get("timeline", [])
    summary = schedule.get("summary", {})

    bookings = store.list_bookings(
        service.db_path(),
        doctor_id=doctor["id"],
        date=today,
    )

    booked_count = sum(
        1 for b in bookings
        if b.get("status") == "booked"
    )

    checked_in_count = sum(
        1 for b in bookings
        if b.get("status") == "checked_in"
    )

    in_service_count = sum(
        1 for b in bookings
        if b.get("status") == "in_service"
    )

    completed_count = sum(
        1 for b in bookings
        if b.get("status") == "done"
    )

    slack_val = summary.get("slack_min", 0.0)
    proj_finish = summary.get("projected_finish_12h") or summary.get("projected_finish", "—")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Booked", booked_count, help="Appointments booked for today that have not checked in yet.")

    with col2:
        st.metric("Checked In", checked_in_count, help="Patients currently waiting in the live queue.")

    with col3:
        st.metric("In Consultation", in_service_count, help="Patient currently being seen.")

    with col4:
        st.metric("Completed", completed_count, help="Completed consultations today.")

    with col5:
        slack_prefix = "+" if float(slack_val) > 0 else ""
        st.metric(
            "Slack",
            f"{slack_prefix}{float(slack_val):.1f} min",
            help="Difference between shift end and projected finish. Negative indicates schedule overrun.",
        )

    with col6:
        st.metric(
            "Projected Finish",
            str(proj_finish),
            help="Estimated finish time of active queue workload.",
        )

    if float(slack_val) < 0:
        st.warning(
            f"⚠️ Schedule Overrun Alert: Active queue workload exceeds shift hours by {abs(float(slack_val)):.1f} minutes. "
            f"Projected finish ({proj_finish}) is beyond shift end ({doctor.get('shift_end')})."
        )
    elif summary.get("shift_ended"):
        st.info(
            f"ℹ️ Doctor shift for today has concluded ({doctor.get('shift_start')} – {doctor.get('shift_end')})."
        )

    st.divider()

    st.markdown("### 📋 Current Live Queue")

    active_timeline = [item for item in timeline if item.get("status") in ("in_service", "checked_in", "booked")]

    if not active_timeline:
        st.info("No active appointments in today's live queue.")
    else:
        table_rows = []
        for item in active_timeline:
            st_val = item.get("status")
            if st_val == "in_service":
                pos_str = "Current Patient"
                wait_str = "In Progress"
                st_label = "🩺 In Consultation"
            elif st_val == "checked_in":
                pos_str = item.get("queue_position", f"#{item.get('queue_pos_num', 1)}")
                w_min = item.get("est_wait_min", 0.0)
                wait_str = f"{float(w_min):.0f} min" if w_min is not None else "0 min"
                st_label = "✅ Checked In"
            else:  # booked
                pos_str = "--"
                wait_str = f"Scheduled {item.get('est_turn', item.get('planned_start', '—'))}"
                st_label = "📅 Booked"

            table_rows.append({
                "Patient": f"{'🚨 ' if item.get('priority_level', 0) >= 1 else ''}{item.get('patient_name', 'Patient')}",
                "Status": st_label,
                "Queue Position": pos_str,
                "Est. Wait": wait_str,
                "Est. Turn": item.get("est_turn", "—"),
                "Priority": "🚨 Emergency" if item.get("priority_level", 0) >= 1 else "Routine",
            })
        st.dataframe(table_rows, hide_index=True, width="stretch")

    if timeline:
        st.markdown("#### Patient Actions & Details")
        for item in timeline:
            booking_id = item.get("id")
            status = item.get("status", "booked")
            emergency = item.get("priority_level", 0) >= 1

            title = (
                f"{'🚨 ' if emergency else ''}"
                f"{item.get('patient_name', 'Patient')} "
                f"— {_status_badge(status)}"
                f"{' (' + item.get('queue_position') + ')' if item.get('queue_position') and status in ('checked_in', 'in_service') else ''}"
            )

            with st.expander(title):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.write(
                        "**Planned Slot:**",
                        scheduling.format_clock_12h(item.get("planned_start") or item.get("slot_start", "—")),
                    )
                    st.write(
                        "**Estimated Turn:**",
                        item.get("est_turn", "—"),
                    )

                with c2:
                    if status == "checked_in":
                        w_min = item.get("est_wait_min", 0.0)
                        st.write(
                            "**Estimated Wait:**",
                            f"{float(w_min):.0f} min" if w_min is not None else "0 min",
                        )
                        st.write(
                            "**Queue Position:**",
                            item.get("queue_position", "—"),
                        )
                    elif status == "in_service":
                        st.write(
                            "**Queue Position:**",
                            "Current Patient",
                        )
                        st.write(
                            "**Estimated Turn:**",
                            "Now",
                        )
                    else:
                        st.write(
                            "**Queue Position:**",
                            item.get("queue_position", "Available after check-in"),
                        )
                        st.write(
                            "**Status:**",
                            status.replace("_", " ").title(),
                        )

                with c3:
                    if emergency:
                        st.error("🚨 EMERGENCY PRIORITY")
                    else:
                        st.info("Routine priority")
                    st.write(
                        "**Department:**",
                        doctor.get("department", "General"),
                    )

                if item.get("emergency_reason"):
                    st.warning(
                        f"Emergency reason: {item['emergency_reason']}"
                    )

                # Patient action buttons
                if status == "booked":
                    if st.button(
                        "Check In",
                        key=f"checkin_{booking_id}",
                        type="primary",
                    ):
                        try:
                            service.doctor_check_in(
                                user,
                                booking_id,
                            )
                            st.success("Patient checked in and entered live queue.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                elif status == "checked_in":
                    if st.button(
                        "Start Consultation",
                        key=f"start_{booking_id}",
                        type="primary",
                    ):
                        try:
                            service.doctor_start_service(
                                user,
                                booking_id,
                            )
                            st.success("Consultation started.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                elif status == "in_service":
                    st.markdown("### Complete Consultation")

                    duration = st.number_input(
                        "Actual consultation duration (minutes)",
                        min_value=1.0,
                        max_value=180.0,
                        value=15.0,
                        step=1.0,
                        key=f"duration_{booking_id}",
                    )

                    notes = st.text_area(
                        "Clinical consultation notes",
                        key=f"notes_{booking_id}",
                        placeholder=(
                            "Enter the consultation summary, "
                            "observations, or follow-up instructions."
                        ),
                    )

                    if st.button(
                        "Complete Consultation",
                        key=f"complete_{booking_id}",
                        type="primary",
                    ):
                        try:
                            result = service.doctor_complete_service(
                                user,
                                booking_id,
                                service_minutes=duration,
                                notes=notes,
                            )
                            n_new = int(result.get("new_open_slots") or 0)
                            saved = float(result.get("time_saved_min") or 0.0)
                            buf = float(result.get("suggested_buffer_min") or 0.0)
                            msg = "Consultation completed successfully."
                            if result.get("finishing_early") and n_new > 0:
                                msg += (
                                    f" Early finish reclaimed {n_new} new ML slot(s) "
                                    f"from remaining duty time "
                                    f"(≈{saved:.0f} min saved; dynamic buffer {buf:.1f} min)."
                                )
                            elif n_new > 0:
                                msg += f" {n_new} open slot(s) remain in today's duty window."
                            st.success(msg)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                if status in ("booked", "checked_in"):
                    if st.button(
                        "Mark No-Show",
                        key=f"noshow_{booking_id}",
                    ):
                        try:
                            service.doctor_set_status(
                                user,
                                booking_id,
                                "no_show",
                            )
                            st.warning("Patient marked as no-show.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                if not emergency and status in (
                    "booked",
                    "checked_in",
                ):
                    st.markdown("#### Emergency Priority")

                    reason = st.text_input(
                        "Emergency reason",
                        key=f"emergency_reason_{booking_id}",
                    )

                    if st.button(
                        "Assign Emergency Priority",
                        key=f"emergency_{booking_id}",
                    ):
                        if not reason.strip():
                            st.error(
                                "Enter a reason before assigning priority."
                            )
                        else:
                            try:
                                service.assign_emergency_priority(
                                    user,
                                    booking_id,
                                    priority_level=1,
                                    reason=reason,
                                )
                                st.success(
                                    "Emergency priority assigned."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))

    st.divider()

    st.markdown("### 📅 Upcoming Appointments")
    all_b = store.list_bookings(service.db_path(), doctor_id=doctor["id"])
    upcoming = [b for b in all_b if b.get("appointment_date", "") > today and b.get("status") in ("booked", "checked_in")]
    upcoming.sort(key=lambda b: (b.get("appointment_date", ""), b.get("slot_start") or ""))

    if not upcoming:
        st.info("No upcoming future appointments scheduled.")
    else:
        upcoming_rows = []
        for b in upcoming:
            upcoming_rows.append({
                "Patient": f"{'🚨 ' if int(b.get('priority_level') or 0) >= 1 else ''}{b.get('patient_name', 'Patient')}",
                "Date": scheduling.format_date_short(b.get("appointment_date")),
                "Time": scheduling.format_clock_12h(b.get("slot_start")),
                "Department": b.get("department"),
                "Priority": "🚨 Emergency" if int(b.get("priority_level") or 0) >= 1 else "Routine",
                "Status": b.get("status", "booked").replace("_", " ").title(),
            })
        st.dataframe(upcoming_rows, hide_index=True, width="stretch")


def _shift_tab(user, doctor):
    st.subheader("🕐 Publish Duty & ML Timetable")
    st.caption(
        "When you arrive, publish your duty hours for the day. "
        "Slots are packed from the service-time ML model plus a "
        "crowd-aware dynamic buffer — admin does not set your availability."
    )

    duty_date = st.date_input(
        "Duty date",
        value=date.today(),
        key="doctor_duty_date",
    )
    date_iso = duty_date.isoformat()
    duty = store.get_doctor_duty(service.db_path(), doctor["id"], date_iso)
    on_duty = bool(duty and duty.get("is_available"))

    if on_duty:
        st.success(
            f"On duty {scheduling.format_date_human(date_iso)} · "
            f"{duty.get('shift_start') or doctor.get('shift_start')} – "
            f"{duty.get('shift_end') or doctor.get('shift_end')}"
        )
    else:
        st.warning("Not published for this date — patients cannot book until you open duty hours.")

    default_start = (duty or {}).get("shift_start") or doctor.get("shift_start", "09:00")
    default_end = (duty or {}).get("shift_end") or doctor.get("shift_end", "13:00")

    col1, col2 = st.columns(2)
    with col1:
        shift_start = st.time_input(
            "Shift start",
            value=_parse_time(default_start),
            key="doctor_shift_start",
        )
    with col2:
        shift_end = st.time_input(
            "Shift end",
            value=_parse_time(default_end),
            key="doctor_shift_end",
        )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Publish duty & open ML slots", type="primary", key="btn_publish_duty"):
            try:
                service.publish_doctor_duty(
                    user,
                    doctor["id"],
                    date_iso,
                    is_available=True,
                    shift_start=shift_start.strftime("%H:%M"),
                    shift_end=shift_end.strftime("%H:%M"),
                )
                st.success(
                    f"Duty published for {scheduling.format_date_human(date_iso)}. "
                    f"ML slots are now bookable inside your shift window."
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with b2:
        if st.button("Close duty for this date", key="btn_close_duty"):
            try:
                service.publish_doctor_duty(
                    user,
                    doctor["id"],
                    date_iso,
                    is_available=False,
                    shift_start=shift_start.strftime("%H:%M"),
                    shift_end=shift_end.strftime("%H:%M"),
                )
                st.warning("Duty closed — new bookings for this date are blocked.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with st.expander("⚡ Batch Actions (Publish Multiple Dates)", expanded=False):
        st.caption("Conveniently open or close multiple dates using the shift hours set above.")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Publish Today & Tomorrow", key="btn_quick_today_tomorrow"):
                try:
                    for d_offset in [0, 1]:
                        d_str = (date.today() + timedelta(days=d_offset)).isoformat()
                        service.publish_doctor_duty(
                            user, doctor["id"], d_str, is_available=True,
                            shift_start=shift_start.strftime("%H:%M"),
                            shift_end=shift_end.strftime("%H:%M"),
                        )
                    st.success("Today and Tomorrow published! ML slots are now open for patient booking.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with col_b2:
            if st.button("Publish Next 7 Days", key="btn_quick_7_days"):
                try:
                    for d_offset in range(7):
                        d_str = (date.today() + timedelta(days=d_offset)).isoformat()
                        service.publish_doctor_duty(
                            user, doctor["id"], d_str, is_available=True,
                            shift_start=shift_start.strftime("%H:%M"),
                            shift_end=shift_end.strftime("%H:%M"),
                        )
                    st.success("Next 7 days published! ML slots are now open for patient booking.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("### 📋 My Published Open Dates")
    my_pub = service.list_doctor_published_dates(doctor_id=doctor["id"], from_date=date.today().isoformat())
    if my_pub:
        pub_rows = [
            {
                "Date": scheduling.format_date_human(r["available_date"]),
                "Shift Hours": f"{r.get('shift_start', '—')} to {r.get('shift_end', '—')}",
                "Status": "🟢 Open for Booking",
            }
            for r in my_pub
        ]
        st.dataframe(pub_rows, hide_index=True, width="stretch")
    else:
        st.info("You have not opened any upcoming dates yet. Select a date above and click 'Publish duty & open ML slots' to open it for patient booking.")

    st.divider()
    st.markdown("### Dynamic buffer (crowd-aware)")
    try:
        schedule = service.live_schedule(doctor, date_iso) if on_duty else None
        summary = (schedule or {}).get("summary") or {}
        buf = float(summary.get("suggested_buffer_min") or service.get_cfg().get("buffer", {}).get("base_minutes", 10))
        waiting = int(summary.get("waiting_count") or summary.get("checked_in_count") or 0)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Live buffer", f"{buf:.1f} min")
        with c2:
            st.metric("Waiting now", waiting)
        with c3:
            st.metric("Reclaimed slots", int(summary.get("new_open_slots") or 0))
        with c4:
            st.metric("Slack", f"{float(summary.get('slack_min') or 0.0):+.1f} min")
        st.caption(
            "Buffer is dynamic: shrinks when more people are waiting or the day is behind, "
            "and widens when consults overrun. Early finishes auto-open new ML slots in leftover duty time."
        )
        extras = (schedule or {}).get("extra_open_slots") or []
        if extras:
            st.markdown("#### Newly opened slots from early finishes / leftover time")
            st.dataframe(
                [
                    {
                        "Start": scheduling.format_clock_12h(s.get("slot_start") or s.get("start")),
                        "End": scheduling.format_clock_12h(s.get("slot_end") or s.get("end")),
                        "Service (min)": s.get("service_minutes"),
                        "Buffer (min)": s.get("buffer_minutes"),
                        "Origin": s.get("origin", "reclaimed"),
                    }
                    for s in extras
                ],
                hide_index=True,
                width="stretch",
            )
    except Exception as exc:
        st.caption(f"Buffer preview unavailable: {exc}")

    st.divider()
    st.markdown("### ML-Packed Shift Preview")
    if not on_duty:
        st.info("Publish duty hours above to generate the ML timetable for this date.")
        return

    try:
        preview = service.preview_shift_plan(doctor, date_iso)
    except Exception as exc:
        st.error(f"Unable to generate timetable preview: {exc}")
        return

    if not preview:
        st.info(
            "No ML-generated shift slots are available. "
            "Check that the service-time model is trained."
        )
        return

    rows = [
        {
            "Start": scheduling.format_clock_12h(slot.get("slot_start") or slot.get("start")),
            "End": scheduling.format_clock_12h(slot.get("slot_end") or slot.get("end")),
            "Duration (min)": slot.get("duration", slot.get("service_minutes", "—")),
            "Buffer (min)": slot.get("buffer_minutes", "—"),
        }
        for slot in preview
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.metric("Planned Capacity", len(preview))


def _parse_time(value):
    from datetime import time

    try:
        hour, minute = str(value).split(":")[:2]

        return time(
            int(hour),
            int(minute),
        )
    except Exception:
        return time(9, 0)


def _briefing_tab(user, doctor):
    st.subheader("🤖 Gemini Shift Briefing")

    today = date.today().isoformat()

    try:
        schedule = service.live_schedule(
            doctor,
            today,
        )

        bookings = store.list_bookings(
            service.db_path(),
            doctor_id=doctor["id"],
            date=today,
        )

        context = {
            "doctor": doctor,
            "date": today,
            "summary": schedule.get("summary", {}),
            "timeline": schedule.get("timeline", []),
            "bookings": bookings,
        }

        explainer = service.explainer()

        if hasattr(explainer, "explain_live_day"):
            response = explainer.explain_live_day(context)
            st.info(response)
        elif hasattr(explainer, "answer"):
            response = explainer.answer(
                "Provide a factual shift briefing.",
                context,
            )
            st.info(response)
        else:
            st.info(
                "The GenAI explainer is not available. "
                "The live queue data is still available."
            )

    except Exception as exc:
        st.warning(
            f"AI briefing unavailable: {exc}"
        )


def _copilot_tab(user, doctor):
    st.subheader("🧠 Clinical Queue Copilot")

    if "doctor_chat" not in st.session_state:
        st.session_state["doctor_chat"] = []

    c_hist, c_clear = st.columns([5, 1])
    with c_clear:
        if st.button("🗑️ Clear Chat", key="doc_clear_chat"):
            st.session_state["doctor_chat"] = []
            st.rerun()

    clicked_question = None
    with st.expander("💡 Suggested questions", expanded=False):
        suggestions = [
            "How many patients are waiting in my queue right now?",
            "What is my projected shift finish time?",
            "Are there any emergency priority patients in my queue?",
            "Do I have any expected no-shows today?",
            "Give me a summary of my clinic session today.",
        ]
        for s in suggestions:
            if st.button(s, key=f"doc_sugg_{s[:25]}"):
                clicked_question = s

    for message in st.session_state["doctor_chat"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    typed_question = st.chat_input("Ask about your live queue...")
    question = typed_question or clicked_question

    if not question:
        return

    st.session_state["doctor_chat"].append({"role": "user", "content": question})

    today = date.today().isoformat()

    try:
        schedule = service.live_schedule(
            doctor,
            today,
        )

        context = {
            "doctor": doctor,
            "date": today,
            "summary": schedule.get("summary", {}),
            "timeline": schedule.get("timeline", []),
        }

        explainer = service.explainer()

        answer = None

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

        if not answer:
            answer = (
                "The clinical AI assistant is currently unavailable. "
                "Please use the live queue data shown in the console."
            )

    except Exception as exc:
        answer = f"Unable to process the request: {exc}"

    st.session_state["doctor_chat"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.rerun()


def render_doctor_portal(user):
    """Render the complete doctor portal."""

    service.require_role(
        user,
        ("doctor",),
    )
    service.auto_handle_expired_appointments()

    doctor_id = user.get("doctor_id")

    if not doctor_id:
        st.error(
            "This doctor account is not linked to a doctor roster record."
        )
        return

    doctor = service.get_doctor(
        doctor_id,
    )

    if not doctor:
        st.error("Doctor roster record not found.")
        return

    top1, top2 = st.columns([5, 1])

    with top1:
        st.title("👨‍⚕️ Doctor Clinical Console")
        st.caption(
            f"{doctor.get('name')} · "
            f"{doctor.get('department')}"
        )

    with top2:
        if st.button("Logout"):
            _logout()

    tabs = st.tabs(
        [
            "📋 Live Queue",
            "🕐 Shift & Timetable",
            "🤖 Shift Briefing",
            "🧠 Clinical Copilot",
        ]
    )

    with tabs[0]:
        _live_queue(user, doctor)

    with tabs[1]:
        _shift_tab(user, doctor)

    with tabs[2]:
        _briefing_tab(user, doctor)

    with tabs[3]:
        _copilot_tab(user, doctor)