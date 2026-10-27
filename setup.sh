#!/usr/bin/env bash
# ---------------------------------------------------------------------
# One-shot setup for the Queue Management System.
# Creates a virtual environment, installs dependencies, and prints
# the commands to prepare data, train models, evaluate, and run the app.
# ---------------------------------------------------------------------
set -e

echo ">> Creating virtual environment (.venv) ..."
python3 -m venv .venv

echo ">> Activating and installing dependencies ..."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .   # install the qms package (editable)

echo ""
echo "=========================================================="
echo " Setup complete. Next steps:"
echo "=========================================================="
echo "  1) Prepare data:   python scripts/01_prepare_data.py"
echo "  2) Train models:   python scripts/02_train_models.py"
echo "  3) Evaluate:       python scripts/03_run_evaluation.py"
echo "  4) Launch app:     streamlit run app/streamlit_app.py"
echo ""
echo "  (Optional) Enable GenAI: cp .env.example .env  &&  edit GEMINI_API_KEY"
echo "=========================================================="
