# Panel Speaker Notes — Part 4: GenAI Layer, Web Dashboard & Persistence

**Module owner:** Member 4 (Full-Stack & GenAI Lead) — 1,480 LOC (Python & SQL), ~50 hrs
**Files:** `src/qms/genai/explainer.py`, `app/service.py`, `app/store.py`, `app/streamlit_app.py`, `app/pages/*`

---

## 30-second opening (say this first)

> "Parts 1–3 produce the intelligence — the predictions, the optimised schedule,
> the simulation. **Part 4 is the last mile:** it turns those raw numbers into
> something a patient or a nurse actually uses, wraps it in a web app, and stores
> everything reliably. Three pillars — a **GenAI explanation layer**, a **6-page
> Streamlit dashboard**, and an **ACID-compliant SQLite persistence layer**."

---

## Pillar 1 — GenAI explanation layer  (`src/qms/genai/explainer.py`)

- **What:** turns computed numbers (no-show %, wait, buffer rec, policy scores)
  into a short plain-English rationale for patients/staff. Mirrors the base paper
  (Lim & Yu, 2025): numerical optimiser + Gemini-generated explanation.
- **Never hard-fails** — the headline property:
  - Key + SDK present → **live Gemini 2.0 Flash**.
  - No key / offline / API error → **deterministic template from the same numbers**.
  - `mode` reports `"gemini"` or `"offline-template"`.
  - Model fallback chain: `2.0-flash → 1.5-flash → 1.5-pro` (temp 0.4, ≤400 tokens).
- **Grounded / anti-hallucination** (say this):
  - Every prompt: *"Use ONLY these facts; do not invent numbers or names."*
  - The offline fallback is built from the *same* facts → always consistent.
  - **"The model does the wording, not the maths."**
- **7 explanation surfaces:** `explain_no_show`, `explain_arrival`, `explain_buffer`,
  `explain_schedule`, `explain_wait_estimate`, `explain_live_day`, `answer()` (grounded Q&A).
- **Security:** `GEMINI_API_KEY` from env var or `.env` only — never hardcoded; `.env` git-ignored.

## Pillar 2 — Streamlit web dashboard  (`app/service.py` + 6 pages)

- **6 pages:** Book Appointment · Live Queue · Admin Dashboard · Model Insights ·
  Doctors & Shifts · GenAI Assistant. Cloud-deployable, no on-site hardware.
- **`service.py` = the bridge/adapter:**
  - Puts `src/` on the path → `import qms` works with no install.
  - Loads config + trained model bundles + eval artifacts (all optional).
  - Builds **complete feature rows**: dataset medians/modes overridden by form inputs
    (5 form fields → full 20-feature model input).
  - Exposes `predict_no_show`, `predict_wait`, `predict_service`,
    `predict_arrival_delay`, the explainer, and roster/slot functions.
- **Graceful degradation** (talking point):
  - Models not trained → shows setup instructions, doesn't crash.
  - No Gemini key → offline explanations.
  - Hot-reloads a bundle when its `.pkl` mtime changes.
  - Importable outside Streamlit (the smoke test drives it) → UI cleanly separated from logic.

## Pillar 3 — SQLite persistence  (`app/store.py`)

- **Dependency-free** (built-in `sqlite3`). Two tables: `doctors` (roster) + `appointments` (bookings).
- **Lifecycle state machine (the core):**
  `booked → checked_in → in_service → done`  (branches: `no_show`, `cancelled`).
  Each transition stamps a real timestamp (`check_in_time`, `service_start`,
  `service_minutes`) — **these feed Part 2's live re-allocation**, so the DB is the
  ground-truth clock for the whole live schedule, not just storage.
- **"ACID-compliant"** is defensible: every write runs in a transaction that commits
  only on success and always closes → an error mid-booking leaves the DB untouched,
  never half-written.
- **Idempotent, non-destructive migrations:** `_migrate` adds new columns to old DBs
  on every startup (`ALTER TABLE ADD COLUMN`) — existing rows preserved.
- **Returning-patient history** (`list_known_patients`, `patient_profile`) derives
  `previous_no_show_count` from past bookings → feeds the no-show model.

---

## How the three connect (say this to tie it together)

> "Book a patient — that's a row in SQLite. `service.py` builds the feature row,
> the no-show model scores it, and the GenAI layer turns that percentage into a
> sentence on the page. As the day runs, the lifecycle timestamps in the database
> feed the live re-allocation, and `explain_live_day` briefs the charge nurse.
> **Persistence → prediction → explanation → UI, one loop.**"

---

## Likely questions + crisp answers

**Q: Gemini API down or no key?**
A: Offline template fallback produces the same explanation from the same numbers.
The app never breaks — deliberate design, not an afterthought.

**Q: Does the LLM hallucinate numbers?**
A: No. Prompts are grounded ("use ONLY these facts") and the fallback is
deterministic. The LLM only phrases values the system computed.

**Q: Why SQLite, not Postgres/MySQL?**
A: Zero-config, serverless, ACID, ideal for a single-clinic deployment and a
capstone. The store is a thin layer — swapping DBs later touches one file.

**Q: Why Streamlit?**
A: Pure-Python multipage app, no separate JS frontend, fast to build a clinical
console, cloud-deployable with no on-site hardware.

**Q: How is the API key kept safe?**
A: Read from env var / `.env` only, never in source; `.env` is git-ignored.

---

## Cheat-sheet (memorise)

| Pillar | File(s) | One-liner |
|---|---|---|
| GenAI layer | `src/qms/genai/explainer.py` | Gemini 2.0 Flash + deterministic offline fallback; grounded, never hard-fails |
| Web dashboard | `app/service.py` + `app/pages/` | 6-page Streamlit console; service.py bridges UI ↔ ML; degrades gracefully |
| Persistence | `app/store.py` | Dependency-free ACID SQLite; booking lifecycle state machine; idempotent migrations |
