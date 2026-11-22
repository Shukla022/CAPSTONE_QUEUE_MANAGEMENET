# 🏥 Multi-Objective AI-Driven Hospital Queue Management System

> **Capstone Project — UE23CS441A | Phase 3**  
> PES University, Bangalore — Computer Science & Engineering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-orange?logo=google)
![ML](https://img.shields.io/badge/ML-GradientBoosting-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Abstract

This system is an **AI-powered hospital queue management platform** that combines four Gradient Boosting machine-learning models (no-show prediction, arrival delay, waiting time, service time) with a multi-objective scheduler, conformal prediction uncertainty bands, Pareto-optimal policy search, counterfactual explanations, and a Gemini 2.0 Flash generative-AI explanation layer. The platform exposes three role-based Streamlit portals — **Patient**, **Doctor**, and **Admin** — backed by a SQLite database, and includes an out-of-distribution evaluation pipeline on a real hospital dataset.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit Web App                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Patient      │  │ Doctor       │  │ Admin                │  │
│  │ Portal       │  │ Portal       │  │ Dashboard            │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └─────────────────┴────────────────────┘               │
│                        ▼  service.py  ▼                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   store.py (SQLite)                        │ │
│  │             scheduling.py (Optimizer)                      │ │
│  └───────────────────┬────────────────────────────────────────┘ │
└──────────────────────┼──────────────────────────────────────────┘
                       ▼
┌─────────────────── src/qms/ Library ───────────────────────────┐
│  data/prepare.py   →   Feature engineering + cleaning          │
│  models/train.py   →   4× GradientBoosting models              │
│  models/predict.py →   Inference + conformal intervals         │
│  optimization/     →   Multi-objective Pareto scheduler        │
│  simulation/       →   Queue simulation + policy comparison    │
│  genai/explainer.py→   Gemini 2.0 Flash natural-language msgs  │
└────────────────────────────────────────────────────────────────┘
                       ▼
┌──────────── 5-Script Pipeline ─────────────────────────────────┐
│  01_prepare_data.py  →  02_train_models.py  →  03_evaluation   │
│  04_novelty_analysis.py  →  05_real_hospital_evaluation.py     │
└────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 👤 Patient Portal
- Book appointments by selecting department and available doctor
- View estimated waiting time with **conformal prediction** uncertainty bands
- Receive **Gemini AI-generated** plain-language explanation of wait estimate
- Track appointment status (Scheduled → Active → Completed)
- View full appointment history

### 🩺 Doctor Portal
- View today's appointment schedule with real-time queue status
- Mark patients as arrived, in-progress, or completed
- Toggle availability on/off (removes from patient booking flow)
- View patient no-show risk per appointment

### 🛡️ Admin Dashboard
- Live queue monitoring across all departments
- System-wide metrics: utilisation, fairness score, peak hours
- Manage doctors — add, activate, assign department & shift
- View ML model accuracy reports and feature importance charts
- Gemini AI policy recommendations

### 🔬 Research Novelties
| # | Novelty | Description |
|---|---------|-------------|
| 1 | **Conformal Prediction** | Distribution-free `α=0.10` prediction intervals on waiting time; guaranteed 90% coverage without Gaussian assumptions |
| 2 | **Pareto Optimisation** | NSGA-II-style multi-objective search over `{waiting ×0.4, utilisation ×0.2, fairness ×0.2, emergency ×0.2}` to find non-dominated scheduling policies |
| 3 | **Counterfactual Explanations** | Minimal-perturbation feature changes that flip the predicted outcome (e.g., "book 2 hours earlier to reduce wait by 8 min") |
| 4 | **Fairness Audit** | Demographic-parity + equalised-odds analysis across priority groups, with per-group confusion matrix reporting |

---

## 📁 Project Structure

```
queue-management-system/
│
├── app/                            # Streamlit application
│   ├── streamlit_app.py            # App entry point & sidebar routing
│   ├── service.py                  # Business logic layer
│   ├── store.py                    # SQLite data access layer
│   ├── scheduling.py               # Live slot optimizer
│   ├── portal_patient.py           # Patient portal UI
│   ├── portal_doctor.py            # Doctor portal UI
│   ├── portal_admin.py             # Admin dashboard UI
│   ├── portal_auth.py              # Authentication (login/register)
│   └── qms.db                      # SQLite database (auto-created)
│
├── config/
│   └── config.yaml                 # Central configuration (all tunables)
│
├── data/
│   ├── raw/
│   │   ├── hospital_dataset_preprocessed.csv   # Synthetic training data (20 000 rows)
│   │   ├── realistic_hospital_testing_dataset_v8.csv  # Real hospital data (OOD test)
│   │   └── README.md
│   └── processed/                  # Auto-generated by script 01
│       ├── clean.csv
│       ├── data_profile.json
│       ├── real_hospital_clean.csv
│       └── real_hospital_profile.json
│
├── models/                         # Trained model bundles (auto-generated by script 02)
│   ├── no_show.pkl
│   ├── arrival_delay.pkl
│   ├── waiting_time.pkl
│   └── service_time.pkl
│
├── reports/
│   ├── metrics/                    # JSON metric reports
│   │   ├── model_metrics.json
│   │   ├── policy_comparison.json
│   │   └── buffer_analysis.json
│   └── figures/                    # PNG charts (auto-generated)
│
├── scripts/                        # Numbered pipeline scripts (run in order)
│   ├── _bootstrap.py               # Adds src/ to Python path
│   ├── 01_prepare_data.py
│   ├── 02_train_models.py
│   ├── 03_run_evaluation.py
│   ├── 04_run_novelty_analysis.py
│   ├── 05_evaluate_on_real_hospital.py
│   └── README.md
│
├── src/qms/                        # Core library (installed as package)
│   ├── config.py                   # Config loader
│   ├── data/
│   │   └── prepare.py              # Data cleaning + feature engineering
│   ├── models/
│   │   ├── train.py                # Model training
│   │   └── predict.py              # Inference + conformal intervals
│   ├── optimization/
│   │   └── scheduler.py            # Multi-objective Pareto scheduler
│   ├── simulation/
│   │   └── simulator.py            # Queue simulation engine
│   └── genai/
│       └── explainer.py            # Gemini 2.0 Flash explainer
│
├── tests/                          # Pytest test suite
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_scheduling.py
│   ├── test_simulation.py
│   ├── test_optimization.py
│   ├── test_genai.py
│   ├── test_novelty.py
│   └── test_real_hospital.py
│
├── .env                            # API keys (git-ignored)
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── setup.sh
```

---

## ⚙️ Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.10 or 3.11 | 3.12+ not tested |
| pip | 23+ | bundled with Python |
| Git | any | for cloning |
| Gemini API key | optional | AI explanations work without it via fallback |

> **Windows users:** Use PowerShell or Command Prompt. All commands below work on Windows, macOS, and Linux.

---

## 🚀 Quick Start (Step-by-Step)

### Step 1 — Get the Project

```bash
# If cloning from Git:
git clone <repository-url>
cd queue-management-system

# Or if already downloaded, just open the folder:
cd "queue-management-system"
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

> You should see `(.venv)` at the start of your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4 — Configure API Key (Optional)

Create a `.env` file in the project root (or edit the existing one):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Without a key:** The app still works. AI explanations fall back to a deterministic template generator. Get a free key at [aistudio.google.com](https://aistudio.google.com).

---

## 🔄 Data Pipeline (Run Scripts in Order)

> Run all scripts from the **project root directory** with the virtual environment active.

### Script 01 — Prepare Data

```bash
python scripts/01_prepare_data.py
```
**What it does:**
- Reads `data/raw/hospital_dataset_preprocessed.csv` (20,000 synthetic rows)
- Validates schema, imputes missing values, engineers features (cyclic time encoding, lag statistics, priority mapping)
- Writes `data/processed/clean.csv` and `data/processed/data_profile.json`

**Expected output:**
```
== Step 1/3: preparing data ==
done.
```

---

### Script 02 — Train ML Models

```bash
python scripts/02_train_models.py
```
**What it does:**
- Trains 4 Gradient Boosting models on 80% of `clean.csv` (time-based split)
- Saves model bundles to `models/*.pkl`
- Saves metrics to `reports/metrics/model_metrics.json`
- Generates calibration and feature-importance plots in `reports/figures/`

**Expected output:**
```
== Step 2/3: training models ==
Training no_show ...
Training arrival_delay ...
Training waiting_time ...
Training service_time ...
done.
```

---

### Script 03 — Run Policy Evaluation

```bash
python scripts/03_run_evaluation.py
```
**What it does:**
- Simulates FCFS vs. Priority vs. Multi-Objective scheduling policies
- Runs crowd-aware dynamic buffer analysis
- Saves `reports/metrics/policy_comparison.json` and `buffer_analysis.json`

---

### Script 04 — Run Novelty Analysis

```bash
python scripts/04_run_novelty_analysis.py
```
**What it does:**
- Computes **conformal prediction** coverage at α=0.10
- Runs **Pareto optimisation** (multi-objective policy search)
- Generates **counterfactual explanations**
- Runs **fairness audit** across priority groups
- Saves all novelty metrics to `reports/metrics/`

---

### Script 05 — Real Hospital OOD Evaluation

```bash
python scripts/05_evaluate_on_real_hospital.py
```
**What it does:**
- Prepares `data/raw/realistic_hospital_testing_dataset_v8.csv` (real hospital data)
- Evaluates the 4 trained models on this out-of-distribution dataset
- Tests generalisation from synthetic training → real hospital test
- Saves `data/processed/real_hospital_clean.csv` and OOD metrics

---

## 🌐 Run the Web Application

```bash
streamlit run app/streamlit_app.py
```

The app opens automatically at **http://localhost:8501**

### Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| **Admin** | `admin` | `admin123` |
| **Doctor** | Register via Admin portal, then login | — |
| **Patient** | Register via Patient portal, then login | — |

> To add a doctor: Log in as Admin → Manage Doctors → Add Doctor (fill department, shift times, username/password).

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop
docker-compose down
```

App available at **http://localhost:8501**

---

## 🧪 Run Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_novelty.py -v
python -m pytest tests/test_real_hospital.py -v

# Run with coverage
python -m pytest tests/ --cov=src/qms --cov-report=term-missing
```

---

## 📊 Key Results

### ML Model Performance (Trained on 16,000 / Tested on 4,000 synthetic rows)

| Model | Metric | Value |
|-------|--------|-------|
| **No-Show Prediction** | AUC-ROC | **0.5856** |
| No-Show Prediction | Recall | **1.00** (all no-shows caught) |
| No-Show Prediction | F1 Score | 0.347 |
| **Service Time** | R² | **0.832** (83.2% variance explained) |
| Service Time | MAE | **2.12 min** |
| **Waiting Time** | R² | **0.207** |
| Waiting Time | MAE | 8.38 min |
| **Arrival Delay** | MAE | 8.37 min |

### Top Feature Importances

| Model | #1 Feature | Importance |
|-------|-----------|-----------|
| No-Show | `priority_level` | 36.3% |
| Arrival Delay | `hour_mean_delay` | 24.1% |
| Waiting Time | `queue_per_doctor` | 42.2% |
| Service Time | `department_General_Medicine` | 23.6% |

### Novelty Results

| Novelty | Result |
|---------|--------|
| Conformal Prediction (α=0.10) | ≥90% empirical coverage verified |
| Pareto Scheduler vs FCFS | Reduced mean wait; improved fairness score |
| Counterfactual Explanations | Minimum feature-perturbation sets generated |
| Fairness Audit | Per-group TPR/FPR parity across 3 priority levels |

---

## 📂 Dataset Description

| Dataset | Rows | Purpose |
|---------|------|---------|
| `hospital_dataset_preprocessed.csv` | ~20,000 | **Synthetic** training + validation data |
| `realistic_hospital_testing_dataset_v8.csv` | varies | **Real hospital** OOD evaluation only |

### Key Features Used

| Feature | Description |
|---------|-------------|
| `priority_level` | 1=Emergency, 2=Urgent, 3=Normal |
| `lead_time_hours` | Hours between booking and appointment |
| `no_show_rate` | Patient's historical no-show rate |
| `queue_per_doctor` | Queue length normalised by doctor count |
| `hour_of_day` | Appointment hour (cyclic encoded) |
| `department` | One of 8 departments (Cardiology, ENT, etc.) |

---

## 🏥 Supported Departments

| Department | Department |
|-----------|-----------|
| General Medicine | Cardiology |
| Orthopedics | Gynecology |
| Dermatology | Ophthalmology |
| ENT | Dentistry |

---

## 🤖 GenAI Integration

The system uses **Google Gemini 2.0 Flash** to generate:
- Plain-language wait time explanations for patients ("Your estimated wait is 12–18 minutes because...")
- Policy recommendation summaries for admin
- Counterfactual suggestions ("If you book on Tuesday instead of Monday, your wait drops by ~6 min")

**Fallback:** If `GEMINI_API_KEY` is not set or the API is unavailable, a deterministic template-based explainer activates automatically with no errors.

---

## 👥 Team

| Name | USN | Role |
|------|-----|------|
| **Priyanka** | UE23CS441A | ML Models + Conformal Prediction |
| **Shukla** | UE23CS441A | GenAI Layer + Admin Dashboard |
| **Siddarth** | UE23CS441A | Optimization + Simulation |
| **Sinchana** | UE23CS441A | Patient & Doctor Portals + Database |

---

## 🔧 Configuration Reference

All system parameters are in [`config/config.yaml`](config/config.yaml):

```yaml
scheduler:
  objective_weights:
    waiting:     0.40   # minimise average wait
    utilization: 0.20   # maximise doctor utilisation
    fairness:    0.20   # equitable wait times
    emergency:   0.20   # keep emergency waits low

buffer:
  base_minutes: 10         # default buffer per slot
  max_overbook_fraction: 0.30   # max 30% overbooking

genai:
  model: "gemini-2.0-flash"
  max_output_tokens: 400
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: qms` | Run `pip install -e .` from project root |
| App shows no doctors in patient portal | Add a doctor via Admin → Manage Doctors |
| `GEMINI_API_KEY` errors | Ignore — app falls back to template explainer |
| `streamlit: command not found` | Run `pip install streamlit` or activate `.venv` first |
| Port 8501 already in use | `streamlit run app/streamlit_app.py --server.port 8502` |
| SQLite `database is locked` | Stop other Streamlit instances; delete `app/qms.db` to reset |

---

## 📄 License

This project is submitted as a capstone project for academic evaluation. All code is original work by the team unless otherwise cited in comments.

---

*Built with ❤️ using Python · Streamlit · Scikit-learn · Google Gemini 2.0 Flash*
