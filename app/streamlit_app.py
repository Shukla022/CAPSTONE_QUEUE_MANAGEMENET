"""Main entry point and role-based router for the AI Cloud Queue Management System.

Directs users strictly to their authorized portal based on verified session role:
- Patient Portal (app/portal_patient.py)
- Doctor Clinical Console (app/portal_doctor.py)
- Hospital Admin Portal (app/portal_admin.py)
- Multi-Role Login & Registration (app/portal_auth.py)
"""
from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve()
ROOT = next(p for p in _HERE.parents if (p / "config" / "config.yaml").exists())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]

import streamlit as st
import service
import store
import portal_auth
import portal_patient
import portal_doctor
import portal_admin

# Page configuration
st.set_page_config(
    page_title="AI Cloud Queue Management System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Suppress sidebar navigation to enforce strict server-side role isolation
st.markdown(
    """
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 98% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize database and seed authentic doctors/demo accounts
service.seed_default_system()
store.seed_default_users_and_doctors(service.db_path())

# Verify session state
if "user" not in st.session_state:
    st.session_state["user"] = None

current_user = st.session_state.get("user")

# Role-based Routing
if not current_user:
    portal_auth.render_login_screen()
else:
    role = str(current_user.get("role", "")).lower()

    if role == "patient":
        portal_patient.render_patient_portal(current_user)
    elif role == "doctor":
        portal_doctor.render_doctor_portal(current_user)
    elif role == "admin":
        portal_admin.render_admin_portal(current_user)
    else:
        st.error(f"Unknown or unauthorized role: '{role}'. Please sign in again.")
        if st.button("Log out"):
            st.session_state["user"] = None
            st.session_state.pop("portal", None)
            st.rerun()

cfg = service.get_cfg()
st.caption(f"{cfg.get('project_name', 'Queue Management System')} · Multi-Objective AI Optimization · Role-Gated Portals")
