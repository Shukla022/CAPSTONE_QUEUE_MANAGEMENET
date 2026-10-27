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

    model_name: str = "gemini-2.0-flash"
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
        obj = cls(model_name=g.get("model", "gemini-2.0-flash"),
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
            from google import genai
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
        fallback_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
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
        Never reveals raw internal DB fields (patient_user_id, doctor_id, noshow_prob, etc.).
        Directs booking inquiries to the Book Appointment tab.
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

        # 2. Check for booking requests
        booking_keywords = ("book", "booking", "schedule", "reschedule", "appointment for",
                            "new appointment", "reserve", "dermatology", "cardiology",
                            "orthopedics", "general medicine", "gynecology", "ent", "dentistry", "ophthalmology")
        is_booking_request = any(kw in q_lower for kw in booking_keywords)

        if is_booking_request and not ("status" in q_lower or "queue" in q_lower or "when is" in q_lower or "wait" in q_lower):
            booking_guidance = (
                "To book an appointment (including for Dermatology or any clinic specialty), "
                "please navigate to the '📅 Book Appointment' tab above. There you can select your "
                "preferred department, doctor, date, and available time slot, and confirm your booking. "
                "(Appointments cannot be directly booked through this chat interface.)"
            )
            if not self._live or self._client is None:
                return booking_guidance

            prompt = (
                "You are an empathetic, professional Patient Queue Copilot at a healthcare clinic. "
                "The patient is asking about booking an appointment. Warmly instruct them to use the "
                "'📅 Book Appointment' tab in the navigation above to select their department, doctor, "
                "date, and time slot. Emphasize that appointments cannot be booked directly inside this chat. "
                "Do NOT claim an appointment has been booked. Keep your reply concise (2-3 sentences).\n"
                f"Patient Question: {q}\n"
                f"Facts: {booking_guidance}"
            )
            return self._generate(prompt, booking_guidance)

        # 3. Build safe facts summary for general/queue inquiries
        if safe_appt:
            fallback = (
                f"Hello {patient_name}! Here is your current appointment information:\n"
                f"• Doctor: {safe_appt['doctor_name']} ({safe_appt['department']})\n"
                f"• Date & Time: {safe_appt['appointment_date']} at {safe_appt['appointment_time']}\n"
                f"• Status: {safe_appt['status']}\n"
                f"• Queue Position: {safe_appt['queue_position']}\n"
                f"• Estimated Turn: {safe_appt['estimated_turn']}\n"
                f"• Estimated Wait: {safe_appt['estimated_wait']}"
            )
        else:
            fallback = (
                f"Hello {patient_name}, you currently do not have an active appointment scheduled. "
                f"If you would like to book a consultation, please navigate to the '📅 Book Appointment' tab above."
            )

        if not q:
            return fallback

        prompt = (
            "You are an empathetic, helpful hospital Patient Queue Copilot. Answer the patient's "
            "question warmly, accurately, and concisely (2-4 sentences) using ONLY these safe facts:\n"
            f"{fallback}\n"
            "Rules:\n"
            "- If they ask to book an appointment, guide them to the '📅 Book Appointment' tab.\n"
            "- Do not disclose internal database IDs, technical machine-learning terms, or no-show scores.\n"
            "- Do not claim an appointment has been booked unless it appears in the facts above.\n"
            f"Patient Question: {q}"
        )
        return self._generate(prompt, fallback)

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

