# 🏥 Intelligent Multi-Objective AI-Driven Hospital Queue Management System

> **Capstone Project — UE23CS441A**  
> **Department of Computer Science & Engineering | PES University, Bangalore**  
> **Repository:** [https://github.com/Shukla022/CAPSTONE_QUEUE_MANAGEMENET](https://github.com/Shukla022/CAPSTONE_QUEUE_MANAGEMENET)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![Gemini AI](https://img.shields.io/badge/Gemini-2.0%20Flash-4285F4?logo=google)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite)
![Tests](https://img.shields.io/badge/Tests-43%2F43%20Passing-brightgreen?logo=pytest)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Executive Summary & Abstract

Traditional hospital outpatient departments suffer from rigid time-slotting, high patient wait times, frequent no-shows, and schedule overruns that cause physician burnout. 

This project implements an **Intelligent Multi-Objective Queue Management System (QMS)** that replaces static scheduling with an adaptive, machine-learning-driven pipeline:
1. **Four Specialized Machine Learning Models**: Gradient Boosting regressors and classifiers for predicting **No-Show Probability**, **Consultation Service Time**, **Live Waiting Time**, and **Patient Arrival Delay**.
2. **Conformal Prediction Intervals**: Quantifies uncertainty with distribution-free, finite-sample statistical guarantees ($\alpha = 0.10$, providing $\ge 90\%$ empirical coverage).
3. **Multi-Objective Pareto Scheduler**: Jointly optimizes patient wait times, doctor utilization, schedule fairness, and emergency priority.
4. **Dynamic ML Slot Generation & Early-Finish Duty Reclamation**: Shifts generate appointment slots based on model predictions and adaptive buffers. When doctors finish scheduled appointments early, the system automatically recalculates remaining duty time and opens fresh booking slots.
5. **Real-Time Synchronized Portals**: Three distinct, role-based Streamlit interfaces (**Patient**, **Doctor**, and **Hospital Administrator**) providing an end-to-end synchronized clinical workflow:  
   `📅 Booked` ➔ `✅ Checked In` ➔ `🩺 In Consultation` ➔ `✔️ Completed`.
6. **GenAI Plain-Language Explanations**: Powered by **Google Gemini 2.0 Flash** (with resilient template fallback) to provide transparent queue status, waiting time drivers, and counterfactual advice to patients.

---

## 🏗️ Architecture & Component Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   STREAMLIT CLIENT LAYER                               │
│                                                                                        │
│   ┌──────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────┐   │
│   │     PATIENT PORTAL       │  │      DOCTOR PORTAL       │  │  ADMIN DASHBOARD   │   │
│   │  - Live Ticket & Stepper │  │  - Shift Timetable Mgt   │  │  - Hospital KPIs   │   │
│   │  - Clinic Check-In       │  │  - 1-Click Start Consult │  │  - Live Queue Sync │   │
│   │  - Dynamic Slot Booking  │  │  - Active Consult Card   │  │  - Emergency Intake│   │
│   │  - GenAI Explanations    │  │  - Reclaim Early Slots   │  │  - Policy Audit    │   │
│   └────────────┬─────────────┘  └────────────┬─────────────┘  └─────────┬──────────┘   │
└────────────────┼─────────────────────────────┼──────────────────────────┼──────────────┘
                 │                             │                          │
                 ▼                             ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                APPLICATION SERVICE LAYER                               │
│                                                                                        │
│                      service.py  ◄───►  scheduling.py  ◄───►  store.py                 │
│                 (Coordination)        (Dynamic Slots & Sched)     (SQLite DB)          │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INTELLIGENCE CORE                                    │
│                                                                                        │
│   ┌────────────────────────────────┐                 ┌─────────────────────────────┐   │
│   │      4x Gradient Boosting      │                 │     Conformal Prediction    │   │
│   │  • service_time.pkl (R²=0.83)  │                 │  • Nonconformity scores     │   │
│   │  • no_show.pkl (Recall=1.0)    │                 │  • α=0.10 Coverage (≥90%)   │   │
│   │  • waiting_time.pkl            │                 └──────────────┬──────────────┘   │
│   │  • arrival_delay.pkl           │                                │                  │
│   └───────────────┬────────────────┘                                │                  │
│                   │                                                 │                  │
│                   ▼                                                 ▼                  │
│   ┌────────────────────────────────┐                 ┌─────────────────────────────┐   │
│   │   Pareto-Optimal Scheduler     │                 │   Gemini 2.0 Flash Explainer│   │
│   │  • Multi-objective weights     │                 │  • Plain-English reasoning  │   │
│   │  • Dynamic Buffer Adjustment   │                 │  • Counterfactual guidance  │   │
│   │  • Early-Finish Slot Recovery  │                 │  • Deterministic fallback   │   │
│   └────────────────────────────────┘                 └─────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Synchronized Clinical Workflow

```
[ Patient Portal ]                     [ Doctor Live Queue ]                     [ Admin Live Dashboard ]
        │                                       │                                           │
        ▼                                       │                                           │
 1. Patient arrives at clinic                   │                                           │
    Clicks "✅ Check in at clinic"              │                                           │
    Status: ✅ Checked In ─────────────────────►│ Enters Waiting Queue                      │
    Ticket updates to #1, shows ETA             │ (Displayed with "Start Consultation" btn) │ Checked In KPI increments
        │                                       │                                           │
        │                                       ▼                                           │
        │                              2. Doctor clicks "🩺 Start Consultation"             │
        │                                 Status: 🩺 In Consultation                        │
        ◄───────────────────────────────────────┼──────────────────────────────────────────►│ In Consultation KPI increments
 3. Status: 🩺 In Consultation                  │ Active Consultation Card pinned to top:   │ Doctor queue shows:
    Position: "Current Patient"                 │ - Patient Details & Priority              │ "1. Patient — 🩺 In Consultation"
    Turn: "Now"                                 │ - Elapsed duration & Notes text area      │
        │                                       │ - "🏁 End Consultation / Complete" btn    │
        │                                       │                                           │
        │                                       ▼                                           │
        │                              4. Doctor clicks "🏁 End Consultation"               │
        │                                 Status: ✔️ Completed                              │
        ◄───────────────────────────────────────┼──────────────────────────────────────────►│ Completed KPI increments
 5. Status: ✔️ Completed                        │ - Active Consultation card clears         │ In Consultation KPI decrements
    "Consultation Completed. Full               │ - Automatic Early Finish Check:           │ Doctor queue shows:
    summary saved in Medical History"           │   reclaims duty time as new booking slots │ "1. Patient — ✔️ Completed"
```

---

## ✨ Key Features & Capabilities

### 👤 Patient Portal
- **ML-Predicted Booking Slots**: Shows authentic slots based on the doctor's published duty shift and predicted consultation durations.
- **Clinic Self Check-In**: One-click check-in upon arriving at the clinic that immediately places the patient into the doctor's live queue.
- **Live Ticket & 4-Stage Stepper**: Real-time progress through `📅 Booked` ➔ `✅ Checked In` ➔ `🩺 In Consultation` ➔ `✔️ Completed`.
- **Conformal Uncertainty Bands**: Displays calibrated lower and upper bounds on wait time.
- **Natural Language Explanations**: Powered by Gemini 2.0 Flash explaining why the wait is at its current estimate.
- **Medical Records & History**: Comprehensive consultation summary, duration, and diagnosis notes recorded after completion.

### 🩺 Doctor Portal
- **Shift & Timetable Publisher**: Doctors configure and publish their duty hours (`shift_start` to `shift_end`), which triggers the ML scheduler to pack available consultation slots.
- **Live Queue Console**:
  - **KPIs**: Booked, Checked In, In Consultation, Completed, Schedule Slack (overrun warning), and Projected Finish time.
  - **Pinned Active Consultation Card**: Highlights the patient currently in the exam room, elapsed minutes, notes input, and End Consultation button.
  - **Direct Waiting Queue**: Lists waiting patients with priority tags and a 1-click `🩺 Start Consultation` action.
- **Early-Finish Duty Reclamation**: When a doctor completes visits ahead of schedule, the engine calculates remaining shift slack and dynamically generates new booking slots.
- **Emergency Escalation**: Assign emergency priority (`priority_level = 1`) with clinical reasons to immediately bump urgent patients to the front.

### 📊 Hospital Admin Dashboard
- **Real-Time Hospital KPI Grid**:
  - `Active Doctors` on duty
  - `Registered Patients`
  - `Total Appointments` for selected date
  - `Checked In` (Waiting in live queue)
  - `In Consultation` (Currently seeing doctor)
  - `Completed` (Finished consultations)
- **Live Doctor Queue Breakdown**: Expander per on-duty doctor showing current status (`🩺 In Consultation`, `✅ Checked In`, `✔️ Completed`).
- **Emergency Walk-in Dispatcher**: Immediately assign unscheduled trauma/acute patients to available doctors with automatic queue prioritization.
- **Policy Comparison & Pareto Audit**: Compares scheduling policies (FIFO, Shortest Job First, Emergency First, Multi-Objective Pareto) on wait time, fairness, and doctor idle time.

---

## 🔑 Default User Accounts & Credentials

All default accounts are pre-seeded in the SQLite database ([`app/qms.db`](app/qms.db)).

| Role | Username | Password | Details |
|------|----------|----------|---------|
| **Admin** | `admin` | `admin123` | Full administrative control & live hospital overview |
| **Doctor** | `dr_priya` | `doctor123` | Cardiology (`09:00` – `13:00`) |
| **Doctor** | `dr_rajesh` | `doctor123` | Orthopedics (`10:00` – `14:00`) |
| **Doctor** | `dr_ananya` | `doctor123` | Dermatology (`09:00` – `13:00`) |
| **Doctor** | `dr_sneha` | `doctor123` | Pediatrics (`11:00` – `15:00`) |
| **Doctor** | `dr_vikram` | `doctor123` | General Medicine (`09:00` – `13:00`) |
| **Doctor** | `dr_arjun` | `doctor123` | Neurology (`12:00` – `16:00`) |
| **Doctor** | `dr_kavita` | `doctor123` | Gynecology (`09:00` – `13:00`) |
| **Doctor** | `dr_rohit` | `doctor123` | ENT (`10:00` – `14:00`) |
| **Patient** | `patient_john` | `patient123` | Registered Patient account |
| **Patient** | `patient_alice` | `patient123` | Registered Patient account |
| **Patient** | `patient_bob` | `patient123` | Registered Patient account |

*Note: New patient accounts can also be freely registered via the Patient Portal login screen.*

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Git**
- Optional: Google Gemini API key (for live GenAI explanations)

### 2. Clone the Repository
```bash
git clone https://github.com/Shukla022/CAPSTONE_QUEUE_MANAGEMENET.git
cd CAPSTONE_QUEUE_MANAGEMENET
```

### 3. Create Virtual Environment & Install Dependencies
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

### 4. Configure Environment (Optional)
If using Google Gemini for natural language explanations, set your API key:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key"

# Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key"
```
*(If unset, the application automatically uses high-fidelity deterministic template explanations).*

### 5. Launch the Streamlit Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🧪 Testing & Verification

The project includes unit, regression, smoke, and full end-to-end integration test suites:

```bash
# Run all tests
pytest

# Run end-to-end flow test (verifies Check-In -> Start -> Complete across portals)
pytest tests/test_end_to_end_flow.py -v

# Run ML dynamic slot generation and early finish reclamation tests
pytest tests/test_dynamic_slots_reopen.py -v

# Run with test coverage report
pytest --cov=app --cov=src/qms
```

---

## 📊 Machine Learning Model Benchmarks

Trained on 16,000 synthetic patient visits with 4,000 test records, with out-of-distribution evaluation on real hospital datasets:

| Model | Target Metric | Score | Key Features |
|---|---|---|---|
| **Service Time** (`service_time.pkl`) | $R^2$ / MAE | **0.832** / **2.12 min** | `department`, `hour_of_day`, `active_doctors_count` |
| **No-Show Prediction** (`no_show.pkl`) | AUC / Recall | **0.586** / **1.000** | `priority_level`, `previous_no_show_count`, `lead_time_hours` |
| **Waiting Time** (`waiting_time.pkl`) | MAE | **8.38 min** | `queue_per_doctor`, `slot_arrival_count` |
| **Arrival Delay** (`arrival_delay.pkl`) | MAE | **8.37 min** | `hour_mean_delay`, `dept_mean_delay` |

---

## 🐳 Docker Containerization

To run using Docker:

```bash
# Build and start container
docker-compose up --build

# Run in background
docker-compose up -d

# Stop container
docker-compose down
```
Access the application at `http://localhost:8501`.

---

## 📁 Repository Structure

```
CAPSTONE_QUEUE_MANAGEMENET/
├── .streamlit/
│   └── config.toml                  # Streamlit server & theme configuration
├── app/
│   ├── portal_admin.py              # Hospital Admin console & KPI metrics
│   ├── portal_doctor.py             # Doctor console (active consult & 1-click start)
│   ├── portal_patient.py            # Patient live ticket & clinic check-in
│   ├── qms.db                       # SQLite production database
│   ├── scheduling.py                # ML slot generation & early finish logic
│   ├── service.py                   # Business logic & state management
│   ├── store.py                     # SQLite CRUD queries & schema migrations
│   └── streamlit_app.py             # Streamlit entry point & authentication
├── config/
│   └── config.yaml                  # System weights, buffers, & GenAI settings
├── models/                          # Serialized ML model artifacts (.pkl)
│   ├── service_time.pkl
│   ├── no_show.pkl
│   ├── waiting_time.pkl
│   └── arrival_delay.pkl
├── src/qms/                         # Core Python library
│   ├── data/preprocess.py
│   ├── genai/explainer.py
│   ├── models/train.py
│   ├── optimization/scheduler.py
│   └── simulation/queue.py
├── tests/                           # Pytest regression & integration suite
│   ├── test_end_to_end_flow.py      # Check-In -> Start -> Complete sync test
│   ├── test_dynamic_slots_reopen.py # Dynamic slot creation & reclamation test
│   ├── test_smoke.py
│   ├── test_regression_and_emergency.py
│   └── test_portals_and_features.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 👥 Capstone Project Team

**PES University, Bangalore**  
**Department of Computer Science & Engineering**  
**Course Code:** UE23CS441A (Capstone Project Phase 3)

| Name | Role & Core Contributions |
|---|---|
| **Shukla** | GenAI Layer (Gemini 2.0 Flash integration), Admin Dashboard, & System Architecture |
| **Priyanka** | Machine Learning Pipelines & Conformal Prediction Calibration |
| **Siddarth** | Multi-Objective Optimization, Pareto Scheduler, & Queue Simulation |
| **Sinchana** | Patient & Doctor Clinical Portals, Database Schema, & UI/UX |

---

## 📜 License
This project is developed and released under the **MIT License**.
