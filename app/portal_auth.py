"""Authentication and registration UI for the Queue Management System."""

from __future__ import annotations

import streamlit as st
import service


def _logout():
    st.session_state["user"] = None
    st.session_state.pop("portal", None)
    st.rerun()


def _login_form(role_name: str, expected_role: str):
    st.subheader(f"🔐 {role_name} Sign In")

    username = st.text_input("Username", key=f"login_username_{expected_role}")
    password = st.text_input(
        "Password",
        type="password",
        key=f"login_password_{expected_role}",
    )

    if st.button(f"Sign In as {role_name}", type="primary", use_container_width=True, key=f"btn_signin_{expected_role}"):
        if not username.strip() or not password:
            st.warning("Please enter both username and password.")
            return

        user, error = service.authenticate_user(
            username.strip(),
            password,
            expected_role=expected_role,
        )

        if error:
            st.error(error)
            return

        st.session_state["user"] = user
        st.session_state["portal"] = user["role"]
        st.success(f"Welcome, {user.get('full_name', user['username'])}!")
        st.rerun()


def _register():
    st.subheader("📝 Create Patient Account")

    full_name = st.text_input("Full name", key="reg_name")
    username = st.text_input("Username", key="reg_username")
    email = st.text_input("Email", key="reg_email")
    phone = st.text_input("Phone", key="reg_phone")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=18,
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Prefer not to say", "Female", "Male", "Other"],
        )

    password = st.text_input(
        "Password",
        type="password",
        key="reg_password",
    )

    confirm_password = st.text_input(
        "Confirm password",
        type="password",
        key="reg_confirm",
    )

    if st.button(
        "Create Patient Account",
        type="primary",
        use_container_width=True,
    ):
        if not full_name.strip():
            st.error("Please enter your full name.")
            return

        if password != confirm_password:
            st.error("Passwords do not match.")
            return

        if len(password) < 4:
            st.error("Password must contain at least 4 characters.")
            return

        user_id, error = service.register_user(
            {
                "username": username.strip(),
                "password": password,
                "role": "patient",
                "full_name": full_name.strip(),
                "email": email.strip(),
                "phone": phone.strip(),
                "age": int(age),
                "gender": gender,
            }
        )

        if error:
            st.error(error)
            return

        st.success("Account created successfully.")
        st.info("You can now sign in using your new username and password.")


def render_login_screen():
    """Render the unauthenticated login/registration screen."""

    st.title("🏥 AI Cloud Queue Management System")
    st.caption(
        "Smart appointment booking · Live queue management · "
        "ML predictions · AI assistance"
    )

    st.divider()

    role = st.selectbox(
        "Select Sign-in Role",
        ["Patient", "Doctor", "Admin"],
        key="login_role_selector",
    )

    if role == "Patient":
        tab_login, tab_register = st.tabs(
            ["🔐 Patient Sign In", "📝 Patient Registration / Sign Up"]
        )
        with tab_login:
            _login_form("Patient", "patient")
        with tab_register:
            _register()

    elif role == "Doctor":
        _login_form("Doctor", "doctor")

    elif role == "Admin":
        _login_form("Admin", "admin")