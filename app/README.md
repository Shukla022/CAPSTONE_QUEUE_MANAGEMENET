# `app/`

The Streamlit web app — the cloud-deployable front end. No on-site hardware needed.

```bash
streamlit run app/streamlit_app.py
```

## Pages

| Page | What it does |
|---|---|
| **Book Appointment** | Department → date → **on-duty doctor** → **ML-packed clock-time slot**. Slots open only after the doctor publishes duty hours; widths come from the service-time model plus a **crowd-aware dynamic buffer**. Scores **no-show risk** at booking time with drivers + a GenAI explanation. |
| **Live Queue** | Per-doctor **live schedule**: planned vs. projected times. Completing a consult early **auto-reclaims** leftover duty time as new open slots; buffer re-sizes from queue pressure, no-show risk, and overrun. |
| **Admin Dashboard** | Policy comparison, buffer KPIs, live queues. Admin manages roster/credentials — **not** day-by-day doctor availability. |
| **Doctor portal** | Doctor **publishes duty hours** for a date (opens ML slots), runs the live queue, and sees dynamic buffer + reclaimed slots. |

## Files

- `streamlit_app.py` — landing page + status strip; pages live in `pages/`.
- `service.py` — bridges the UI to the `qms` package: loads config/models/eval
  artifacts, builds complete feature rows (dataset medians overridden by form
  inputs), and exposes prediction + explanation helpers. Importable outside
  Streamlit (the smoke test uses it).
- `store.py` — dependency-free **SQLite** store (`app/qms.db`, git-ignored) for the
  `doctors` roster and the booking lifecycle (bookings pin a `doctor_id`,
  `slot_start`/`slot_end`, and record `service_start`/`service_minutes`).
- `scheduling.py` — pure-function **slot engine** (no Streamlit/DB imports, so it's
  unit-tested directly): `plan_slots` derives a doctor's daily slots from duty hours
  + slot width + buffer; `available_slots` removes taken/past slots; `live_schedule`
  rolls the day forward from real progress to project finish time, reclaim freed
  slots, suggest a buffer, and size safe overbooking headroom.

## Notes

- The app **degrades gracefully**: if models aren't trained yet it shows setup
  instructions instead of crashing; if there's no Gemini key, explanations use the
  offline template.
- Requires the `app` extra: `pip install -e ".[app]"` (Streamlit).
- The booking database resets if you delete `app/qms.db`; the Admin Dashboard also
  has a "delete all bookings" control.
