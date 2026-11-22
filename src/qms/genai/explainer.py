"""Generative-AI explanation layer (with a deterministic offline fallback).

Mirrors the base paper (Lim & Yu, 2025), which pairs a numerical optimiser with a
Gemini-generated natural-language explanation. Here the same idea is applied to
queue decisions: given the *numbers* our models and scheduler produce, this layer
turns them into a short, plain-English rationale a patient or an administrator can
actually read.

Key property: **it never hard-fails**. If ``GEMINI_API_KEY`` is set and a Gemini
SDK is installed, explanations are generated live. Otherwise -- offline, no key,
or any API error -- it falls back to a deterministic template built from the
exact same numbers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from ..config import ROOT, load_config


def _parse_env_file(path: Path) -> Optional[str]:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    except OSError:
        return None
    return None


def load_api_key() -> Optional[str]:
    """Read the key from the environment, then project / cwd ``.env`` files."""
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()
    for path in (ROOT / ".env", Path.cwd() / ".env"):
        found = _parse_env_file(path)
        if found:
            return found
    return None


@dataclass
class GenAIExplainer:
    """Explains queue decisions in natural language, live or offline."""

    model_name: str = "gemini-3.8-flash"
    max_output_tokens: int = 400
    api_key: Optional[str] = None
    _client: object = None
    _sdk: str = ""
    _live: bool = False
    last_error: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self._client:
            self._try_init(api_key=self.api_key)

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "GenAIExplainer":
        cfg = cfg or load_config()
        g = cfg.get("genai", {})
        obj = cls(model_name=g.get("model", "gemini-3.8-flash"),
                  max_output_tokens=int(g.get("max_output_tokens", 400)))
        return obj

    def set_api_key(self, key: str) -> None:
        """Update API key at runtime and re-initialize SDK."""
        clean_key = (key or "").strip()
        if clean_key:
            os.environ["GEMINI_API_KEY"] = clean_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)
        self.api_key = clean_key if clean_key else None
        self._try_init(api_key=clean_key)

    def set_model(self, model: str) -> None:
        """Update model name at runtime."""
        if model and model.strip():
            self.model_name = model.strip()
            self._try_init(api_key=self.api_key)

    def _try_init(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else load_api_key()
        if not key:
            self._live = False
            self._client = None
            self._sdk = ""
            self.last_error = "No GEMINI_API_KEY in the environment or .env file."
            return
        # Newer official SDK
        try:  # pragma: no cover - optional dependency
            import importlib
            genai = importlib.import_module("google.genai")
            self._client = genai.Client(api_key=key)
            self._sdk = "google.genai"
            self._live = True
            self.last_error = ""
            return
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
        # Legacy package listed in pyproject optional extra ``genai``
        try:  # pragma: no cover - optional dependency
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._client = genai.GenerativeModel(self.model_name)
            self._sdk = "google.generativeai"
            self._live = True
            self.last_error = ""
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self._sdk = ""
            self._live = False
            self.last_error = (
                f"{exc}. Install a Gemini SDK: pip install google-genai "
                "or pip install google-generativeai"
            )

    @property
    def mode(self) -> str:
        return "gemini" if self._live else "offline-template"

    def test_connection(self) -> tuple[bool, str]:
        """Validate live API connection with a lightweight prompt."""
        if not self._live or self._client is None:
            return False, f"Offline mode: {self.last_error or 'No API key or SDK configured.'}"
        test_prompt = "Reply with 'OK' if you can read this."
        res = self._generate(test_prompt, grounded_fallback="[OFFLINE_FALLBACK]")
        if res == "[OFFLINE_FALLBACK]":
            return False, self.last_error or "API request failed or returned empty response."
        return True, f"Connected to Gemini ({self.model_name} via {self._sdk}): '{res}'"

    def _generate(self, prompt: str, grounded_fallback: str) -> str:
        """Return Gemini text if live, else the deterministic fallback."""
        if not self._live or self._client is None:
            return grounded_fallback

        # Primary model followed by fallback alternatives
        fallback_models = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
        models_to_try = [self.model_name] + [m for m in fallback_models if m != self.model_name]

        last_exc = None
        for m_name in models_to_try:
            try:  # pragma: no cover - network path
                if self._sdk == "google.genai":
                    resp = self._client.models.generate_content(
                        model=m_name,
                        contents=prompt,
                        config={
                            "max_output_tokens": self.max_output_tokens,
                            "temperature": 0.4,
                        },
                    )
                else:
                    if hasattr(self._client, "model_name") and self._client.model_name != m_name:
                        import google.generativeai as genai
                        client = genai.GenerativeModel(m_name)
                    else:
                        client = self._client
                    resp = client.generate_content(
                        prompt,
                        generation_config={
                            "max_output_tokens": self.max_output_tokens,
                            "temperature": 0.4,
                        },
                    )
                text = (getattr(resp, "text", "") or "").strip()
                if text:
                    self.last_error = ""
                    return text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        if last_exc:
            self.last_error = str(last_exc)
        return grounded_fallback

    def _staff_prompt(self, job: str, fallback: str) -> str:
        return (
            "You are a hospital scheduling assistant. " + job + " "
            "Use ONLY these facts; do not invent numbers or names.\n"
            f"Facts: {fallback}"
        )

    # -- explanation surfaces ------------------------------------------
    def explain_no_show(self, ctx: Dict) -> str:
        prob = float(ctx.get("probability", 0.0))
        risk = "high" if prob >= 0.5 else ("elevated" if prob >= 0.3 else "low")
        article = "an" if risk == "elevated" else "a"
        drivers = ctx.get("drivers", [])
        driver_txt = "; ".join(drivers) if drivers else "recent booking history and slot timing"
        fallback = (
            f"This appointment has {article} {risk} no-show risk (estimated {prob:.0%}). "
            f"The main contributing factors are: {driver_txt}. "
            f"{'Consider a reminder call and keeping a controlled overbooking buffer for this slot.' if prob >= 0.3 else 'No special action needed; standard reminders suffice.'}"
        )
        return self._generate(self._staff_prompt(
            "In 2-3 sentences, explain to staff why an appointment is flagged "
            "and what to do.", fallback), fallback)

    def explain_arrival(self, ctx: Dict) -> str:
        delay = float(ctx.get("delay_min", 0.0))
        slot = ctx.get("slot_start") or "the booked time"
        if delay > 1:
            when = f"about {delay:.0f} minutes after {slot}"
            tip = "Keep a short buffer after this slot in case they check in late."
        elif delay < -1:
            when = f"about {-delay:.0f} minutes before {slot}"
            tip = "They are likely early; they can wait if the previous consult overruns."
        else:
            when = f"close to {slot}"
            tip = "Treat this as an on-time arrival for planning."
        fallback = (
            f"Predicted arrival is {when} (model offset {delay:.0f} min vs the slot). "
            f"{tip} This is a typical offset, not a guaranteed clock time."
        )
        return self._generate(self._staff_prompt(
            "In 2 sentences, tell staff when this patient is likely to arrive.",
            fallback), fallback)

    def explain_buffer(self, rec: Dict) -> str:
        from ..optimization.buffer import DynamicBuffer
        fallback = DynamicBuffer.explain(rec)
        return self._generate(self._staff_prompt(
            "In 2 sentences, explain this overbooking recommendation to an administrator.",
            fallback), fallback)

    def explain_schedule(self, metric_rows: Dict[str, Dict], best: str) -> str:
        best_m = metric_rows.get(best, {})
        base = metric_rows.get("FCFS", {})
        saved = ""
        if base and best_m and base.get("avg_wait") is not None:
            delta = base["avg_wait"] - best_m["avg_wait"]
            if delta > 0:
                saved = (f" versus first-come-first-served it cuts average waiting "
                         f"by {delta:.1f} min (from {base['avg_wait']:.1f} to "
                         f"{best_m['avg_wait']:.1f} min)")
        fallback = (
            f"The '{best}' policy gives the best overall balance"
            f" (composite score {best_m.get('composite_score', 0):.2f}){saved}. "
            f"It holds emergency waiting at {best_m.get('emergency_wait', 0):.1f} min "
            f"while keeping doctor utilisation at {best_m.get('utilization', 0):.0%} "
            f"and fairness at {best_m.get('fairness', 0):.0%}."
        )
        return self._generate(self._staff_prompt(
            "In 3-4 sentences, summarise why this scheduling policy was chosen "
            "for a clinic manager.", fallback), fallback)

    def explain_wait_estimate(self, ctx: Dict) -> str:
        wait = float(ctx.get("predicted_wait", 0.0))
        qlen = ctx.get("queue_length")
        docs = ctx.get("active_doctors")
        name = ctx.get("patient_name")
        who = f"{name}, your" if name else "Your"
        detail = []
        if qlen is not None:
            detail.append(f"{int(qlen)} patient(s) ahead")
        if docs is not None:
            detail.append(f"{int(docs)} doctor(s) on duty")
        detail_txt = (", with " + " and ".join(detail)) if detail else ""
        fallback = (
            f"{who} estimated wait is about {wait:.0f} minutes{detail_txt}. "
            f"This is a live estimate and updates as the queue moves."
        )
        prompt = (
            "You greet a patient who just checked in. In 1-2 warm, plain sentences, "
            f"tell them their wait, using ONLY these facts (no invented numbers).\nFacts: {fallback}"
        )
        return self._generate(prompt, fallback)

    def explain_live_day(self, ctx: Dict) -> str:
        name = ctx.get("doctor_name") or "The doctor"
        date = ctx.get("date") or "today"
        booked = int(ctx.get("booked_count") or 0)
        done = int(ctx.get("done_count") or 0)
        extra = int(ctx.get("new_open_slots") or 0)
        over = int(ctx.get("safe_overbook") or 0)
        finish = ctx.get("projected_finish") or "—"
        slack = float(ctx.get("slack_min") or 0.0)
        ns = ctx.get("expected_no_shows")
        ns_txt = f" Expected no-shows among remaining bookings: {ns}." if ns is not None else ""
        if slack < 0:
            pace = (f"The list is running about {abs(slack):.0f} min past shift end "
                    f"({ctx.get('shift_end', '')}).")
        elif extra:
            pace = f"{extra} leftover slot(s) are open before shift end."
        else:
            pace = "Duty time is fully used by the current list."
        fallback = (
            f"{name}'s list on {date}: {done} of {booked} visits done, "
            f"projected finish {finish}. {pace} Safe extra bookings: +{over}.{ns_txt}"
        )
        return self._generate(self._staff_prompt(
            "In 3 sentences, brief the charge nurse on this doctor's day and "
            "what to do next (reminders, overbook, or slow new bookings).",
            fallback), fallback)

    def explain_patient_copilot(self, question: str, context: Dict) -> str:
        """Patient-facing Queue Copilot: provides safe, warm, plain-language guidance.
        Answers any question about appointments, queue status, available departments,
        clinic hours, check-in procedures, or booking steps. Never reveals raw internal DB fields.
        """
        q = (question or "").strip()
        q_lower = q.lower()
        patient_name = context.get("patient_name") or "there"
        raw_appt = context.get("appointment") or {}

        # 1. Sanitize to strictly patient-safe attributes
        safe_appt = {}
        if isinstance(raw_appt, dict) and raw_appt:
            safe_appt = {
                "doctor_name": raw_appt.get("doctor_name") or "Assigned Doctor",
                "department": raw_appt.get("doctor_department") or raw_appt.get("department") or "Specialty Clinic",
                "appointment_date": raw_appt.get("appointment_date_human") or raw_appt.get("appointment_date") or "Scheduled Date",
                "appointment_time": raw_appt.get("appointment_time_12h") or raw_appt.get("slot_start") or "Scheduled Time",
                "status": str(raw_appt.get("status", "booked")).replace("_", " ").title(),
                "queue_position": str(raw_appt.get("queue_position_display") or raw_appt.get("queue_position") or "Available after check-in"),
                "patients_ahead": raw_appt.get("patients_ahead") if raw_appt.get("patients_ahead") is not None else "N/A",
                "estimated_wait": str(raw_appt.get("live_eta_display") or "Available after check-in"),
                "estimated_turn": str(raw_appt.get("estimated_turn_display") or raw_appt.get("estimated_turn") or raw_appt.get("appointment_time_12h") or "—"),
            }

        # Available departments
        default_depts = [
            "Cardiology", "Dentistry", "Dermatology", "ENT",
            "General Medicine", "Gynecology", "Ophthalmology", "Orthopedics"
        ]
        depts = context.get("available_departments") or default_depts
        dept_str = ", ".join(sorted(depts))

        # Build fallback based on specific user intent
        fallback = self._patient_fallback_answer(q, patient_name, safe_appt, depts)

        # Build comprehensive fact context for Gemini
        facts = {
            "patient_name": patient_name,
            "has_active_appointment": bool(safe_appt),
            "appointment": safe_appt if safe_appt else "No active appointment scheduled",
            "available_departments": depts,
            "clinic_shift_hours": "09:00 - 13:00 (Standard Morning Shift)",
            "booking_instructions": "Patients can book consultations in the '📅 Book Appointment' tab by choosing department, doctor, date, and ML slot.",
            "check_in_instructions": "Patients check in at the clinic or via the '🎫 Live Queue' tab to enter the live queue and get their turn.",
        }
        facts_blob = json.dumps(facts, default=str)

        prompt = (
            "You are an empathetic, professional Patient Queue Copilot at a healthcare hospital. "
            "Answer the patient's question warmly, accurately, and concisely (2-4 sentences) using the facts below.\n\n"
            f"Hospital Facts:\n{facts_blob}\n\n"
            "Rules:\n"
            "- If they say hello, hi, or greet you, greet them warmly by name and ask how you can help.\n"
            "- If they ask what departments/specialties are available, list the available departments clearly.\n"
            "- If they ask to book an appointment or ask about booking, guide them to the '📅 Book Appointment' tab.\n"
            "- If they ask about their queue, wait time, or appointment, reference their appointment facts.\n"
            "- If they have no appointment and ask about their wait/status, politely explain they don't have one yet and offer booking steps.\n"
            "- Never disclose internal database IDs, technical machine-learning parameters, or no-show scores.\n"
            f"Patient Question: {q}"
        )
        return self._generate(prompt, fallback)

    def _patient_fallback_answer(self, question: str, patient_name: str, safe_appt: Dict, depts: List[str]) -> str:
        """Grounded fallback for patient copilot when offline."""
        q = question.lower().strip()
        dept_str = ", ".join(sorted(depts))

        # Greeting check
        if q in ("hi", "hello", "hey", "good morning", "good afternoon", "greetings") or q.startswith(("hi ", "hello ")):
            return (
                f"Hello {patient_name}! I am your Hospital Queue Copilot. How can I help you today? "
                "You can ask about your queue status, available departments, clinic hours, or how to book an appointment."
            )

        # Department / specialty inquiry
        if any(w in q for w in ["dept", "department", "specialt", "specialist", "available doctor"]):
            return (
                f"Our hospital offers the following specialty departments: {dept_str}. "
                "You can view available doctors and ML-generated slots in the '📅 Book Appointment' tab."
            )

        # Check-in inquiry
        if any(w in q for w in ["check in", "checkin", "how to check", "arrive"]):
            return (
                "To check in, navigate to the '🎫 Live Queue' tab above and click '✅ Check in at clinic'. "
                "Once checked in, you will enter the doctor's live queue and receive your live queue number and wait time."
            )

        # Booking inquiry
        if any(w in q for w in ["how to book", "how do i book", "schedule an appointment", "new booking", "book appointment", "make an appointment"]):
            return (
                "To book an appointment, click on the '📅 Book Appointment' tab above. "
                "Select your desired department, on-duty doctor, date, and preferred time slot, then click 'Book Appointment'."
            )

        # Queue / wait / appointment status
        if any(w in q for w in ["queue", "wait", "position", "turn", "when is", "my appointment", "status"]):
            if safe_appt:
                return (
                    f"Hello {patient_name}! Your appointment with {safe_appt['doctor_name']} ({safe_appt['department']}) "
                    f"is on {safe_appt['appointment_date']} at {safe_appt['appointment_time']}. "
                    f"Status: {safe_appt['status']} | Queue Position: {safe_appt['queue_position']} | "
                    f"Estimated Wait: {safe_appt['estimated_wait']}."
                )
            return (
                f"Hello {patient_name}, you currently do not have an active appointment scheduled today. "
                "If you would like to schedule a consultation, please visit the '📅 Book Appointment' tab above."
            )

        # General / default fallback
        if safe_appt:
            return (
                f"Hello {patient_name}! Your appointment is confirmed with {safe_appt['doctor_name']} ({safe_appt['department']}) "
                f"at {safe_appt['appointment_time']}. Status: {safe_appt['status']}. "
                "Let me know if you need assistance with your queue or directions."
            )
        return (
            f"Hello {patient_name}! I am your Hospital Queue Copilot. Our hospital features {len(depts)} departments ({dept_str}). "
            "Please visit the '📅 Book Appointment' tab to schedule a consultation, or the '🎫 Live Queue' tab to track your visit."
        )

    def explain_clinical_copilot(self, question: str, context: Dict) -> str:
        """Hospital AI Copilot — answers any operational question from live real-time data.

        Works for both Admin context (hospital_summary, department_load, doctor_wise_load)
        and Doctor context (doctor, summary, timeline). Never invents data.
        """
        q = (question or "").strip()
        if not q:
            return "Please ask a question about today's hospital operations."

        # Detect context type
        is_admin_context = "hospital_summary" in context or "department_load" in context
        is_doctor_context = "doctor" in context or "timeline" in context

        blob = json.dumps(context, default=str)
        # Keep blob under 3000 chars for token budget
        if len(blob) > 3000:
            blob = blob[:3000] + "... [truncated]"

        if is_admin_context:
            fallback = self._admin_fallback_answer(q, context)
            prompt = (
                "You are the Hospital Administrative AI Copilot. You have access to live, real-time "
                "hospital data for today. Answer the administrator's question accurately and concisely "
                "(3-5 sentences) using ONLY the JSON facts below. Do not invent numbers, patient names, "
                "or percentages not present in the facts. If the facts don't contain the answer, say so.\n\n"
                f"Real-Time Hospital Data:\n{blob}\n\n"
                f"Administrator Question: {q}"
            )
        elif is_doctor_context:
            doc = context.get("doctor", {})
            summary = context.get("summary", {})
            timeline = context.get("timeline", [])
            doc_name = doc.get("name", "Doctor") if isinstance(doc, dict) else "Doctor"
            waiting_count = sum(1 for t in timeline if t.get("status") in ("checked_in", "in_service"))
            done_count = sum(1 for t in timeline if t.get("status") == "done")
            fallback = (
                f"Dr. {doc_name}'s queue today: {len(timeline)} total patients, "
                f"{waiting_count} waiting/in-service, {done_count} completed. "
                f"Projected finish: {summary.get('projected_finish_12h') or summary.get('projected_finish', '—')}. "
                f"Slack: {float(summary.get('slack_min') or 0.0):+.1f} min."
            )
            prompt = (
                "You are a Clinical Queue AI Copilot for a doctor. Answer the doctor's question "
                "accurately in 2-4 sentences using ONLY these facts. Do not invent patient names or times.\n\n"
                f"Live Queue Data:\n{blob}\n\n"
                f"Doctor Question: {q}"
            )
        else:
            # Generic admin/staff context
            fallback = f"Live hospital data: {blob[:500]}"
            prompt = (
                "You are the Hospital AI Copilot. Answer this staff question using ONLY "
                "the following real-time facts. Do not invent numbers or names.\n\n"
                f"Facts: {blob}\n\nQuestion: {q}"
            )

        result = self._generate(prompt, fallback if is_doctor_context else self._admin_fallback_answer(q, context))
        return result or fallback

    def _admin_fallback_answer(self, question: str, context: Dict) -> str:
        """Build a plain-text answer from live admin context when Gemini is unavailable."""
        q = question.lower()
        s = context.get("hospital_summary", {})
        dept_load = context.get("department_load", {})
        doctor_load = context.get("doctor_wise_load", [])
        today = context.get("today_date", "today")

        if any(w in q for w in ["waiting", "wait", "queue", "checked in"]):
            return (
                f"Currently {s.get('checked_in_waiting', 0)} patient(s) are checked in and waiting, "
                f"and {s.get('in_service_now', 0)} patient(s) are in active consultation. "
                f"Total appointments today: {s.get('appointments_today', 0)}."
            )
        if any(w in q for w in ["doctor", "duty", "on duty", "available", "who is"]):
            if doctor_load:
                lines = [
                    f"{d['name']} ({d['department']}): {d['total_patients']} patients, "
                    f"{d['completed']} done, {d['remaining_booked']} remaining, shift {d['shift']}"
                    for d in doctor_load
                ]
                return f"Doctors on duty today ({today}):\n" + "\n".join(f"• {l}" for l in lines)
            return f"No doctors on duty yet for today ({today})."
        if any(w in q for w in ["department", "dept", "busiest", "load", "which"]):
            if dept_load:
                sorted_depts = sorted(dept_load.items(), key=lambda x: x[1]["total"], reverse=True)
                lines = [
                    f"{dept}: {v['total']} total, {v['checked_in']} waiting, "
                    f"{v['in_service']} in service, {v['done']} done"
                    for dept, v in sorted_depts
                ]
                return "Department load today:\n" + "\n".join(f"• {l}" for l in lines)
            return "No department data for today yet."
        if any(w in q for w in ["no show", "no-show", "absent", "miss", "rate"]):
            return (
                f"No-shows today: {s.get('no_shows_today', 0)} out of "
                f"{s.get('appointments_today', 0)} appointments "
                f"(rate: {s.get('no_show_rate_today', '0%')})."
            )
        if any(w in q for w in ["complete", "done", "finish"]):
            return (
                f"Completed consultations today: {s.get('completed_today', 0)} out of "
                f"{s.get('appointments_today', 0)} total appointments."
            )
        if any(w in q for w in ["patient", "register", "total"]):
            return (
                f"Total registered patients in the system: {s.get('total_registered_patients', 0)}. "
                f"Appointments today: {s.get('appointments_today', 0)}."
            )
        # Summary / default
        return (
            f"Hospital Status — {today}:\n"
            f"• Doctors on duty: {s.get('doctors_on_duty_today', 0)}\n"
            f"• Total appointments: {s.get('appointments_today', 0)}\n"
            f"• Checked in & waiting: {s.get('checked_in_waiting', 0)}\n"
            f"• In consultation: {s.get('in_service_now', 0)}\n"
            f"• Completed: {s.get('completed_today', 0)}\n"
            f"• No-shows: {s.get('no_shows_today', 0)} ({s.get('no_show_rate_today', '0%')})\n"
            f"• Registered patients: {s.get('total_registered_patients', 0)}"
        )

    def explain_counterfactual(self, scenario: str, cf_data: Dict) -> str:
        """Explains a what-if counterfactual scenario (punctuality, emergency, or doctor staffing)."""
        from .counterfactual import CounterfactualEngine
        factual = float(cf_data.get("factual_value", 0.0))
        counterfactual = float(cf_data.get("counterfactual_value", 0.0))
        delta = float(cf_data.get("delta_minutes", 0.0))
        default_exp = cf_data.get("explanation") or (
            f"Under this counterfactual scenario ({scenario}), the wait time shifts "
            f"from {factual:.1f} min to {counterfactual:.1f} min (delta: {delta:.1f} min)."
        )

        prompt = (
            f"You are the hospital queue optimization explainer. Explain this what-if "
            f"counterfactual analysis in 2 clear, empathetic sentences to a patient or clinic staff:\n"
            f"Scenario: {scenario}\n"
            f"Actual Wait: {factual:.1f} min\n"
            f"Counterfactual Wait: {counterfactual:.1f} min\n"
            f"Difference: {delta:+.1f} min\n"
            f"Details: {default_exp}"
        )
        return self._generate(prompt, default_exp)

    def answer(self, question: str, facts: Dict) -> str:
        """Grounded Q&A: Gemini may only use ``facts``; otherwise a safe readable recap."""
        q = (question or "").strip()
        # If passed patient context with appointment, route through explain_patient_copilot
        if isinstance(facts, dict) and ("patient_name" in facts or "appointment" in facts):
            return self.explain_patient_copilot(question, facts)

        # Remove internal sensitive fields before serializing
        safe_facts = {}
        if isinstance(facts, dict):
            safe_facts = {
                k: v for k, v in facts.items()
                if not any(k.startswith(p) for p in ("patient_user_id", "doctor_id", "password", "noshow_prob"))
            }
        else:
            safe_facts = facts

        blob = json.dumps(safe_facts, default=str)
        fallback = (
            "Offline mode — I can provide the clinic queue numbers and scheduling summaries. "
            f"Active clinic facts: {blob[:400]}"
        )
        if not q:
            return "Ask a question about today's bookings, risk, or capacity."
        prompt = (
            "You are the clinic's queue assistant. Answer the staff question in "
            "2-5 sentences using ONLY the JSON facts. If the facts do not contain "
            "the answer, say so. Do not invent patients, times, or percentages.\n"
            f"Question: {q}\nFacts: {blob}"
        )
        return self._generate(prompt, fallback)


def get_explainer(cfg: dict | None = None) -> GenAIExplainer:
    return GenAIExplainer.from_config(cfg)

