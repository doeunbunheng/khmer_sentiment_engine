"""Khmer Sentiment Engine — Streamlit dashboard (Week 6).

Talks to the FastAPI server (src/api.py) over HTTP — the exact production
pipeline (v2 model + aspects + OOD guard). Includes an unseen-data benchmark
page that sends the 989 held-out kh-polarity rows through the live API.

Run (from the project root, API server must be up first):
    .venv\\Scripts\\streamlit run app\\dashboard.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.api_client import ApiError
from app.dashboard_utils import get_client

st.set_page_config(
    page_title="Khmer Sentiment Engine",
    page_icon=":material/chat:",
    layout="wide",
)

if "auth" not in st.session_state:
    st.session_state.auth = None
if "dashboard_base_url" not in st.session_state:
    st.session_state.dashboard_base_url = "http://127.0.0.1:8000"

auth = st.session_state.auth
client = get_client(st.session_state.dashboard_base_url)

# ---- sidebar: connection status + account ----------------------------------

with st.sidebar:
    st.markdown("### :material/chat: Khmer Sentiment Engine")
    base_url = st.text_input(
        "API server", value=st.session_state.dashboard_base_url
    )
    if base_url.strip():
        st.session_state.dashboard_base_url = base_url.strip()
    try:
        health = client.health()
        st.badge("API online", icon=":material/check_circle:", color="green")
        st.caption(f"model: {health.get('model', '?')}  ·  db: {health.get('db')}")
    except ApiError as exc:
        st.badge("API offline", icon=":material/error:", color="red")
        st.caption(str(exc))

    if auth:
        st.markdown("---")
        st.markdown(f"**{auth['username']}**", text_alignment="left")
        st.badge(auth["role"], icon=":material/person:")
        if st.button("Log out", icon=":material/logout:"):
            st.session_state.auth = None
            st.rerun()

# ---- logged out: login / register ------------------------------------------

if not auth:
    st.title("Khmer Sentiment Engine")
    st.caption(
        "3-class sentiment (positive / negative / neutral) + business aspects "
        "+ emotions for Khmer, English, and mixed comments."
    )
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Log in", text_alignment="left")
            with st.form("login_form"):
                username = st.text_input("Username", max_chars=64)
                password = st.text_input("Password", type="password", max_chars=128)
                if st.form_submit_button(
                    "Log in", type="primary", icon=":material/login:"
                ):
                    if not username or not password:
                        st.error("Enter a username and password.")
                    else:
                        try:
                            data = client.login(username, password)
                        except ApiError as exc:
                            st.error(f":material/error: {exc}")
                        else:
                            st.session_state.auth = {
                                "token": data["token"],
                                "user_id": data["user_id"],
                                "role": data["role"],
                                "username": username,
                            }
                            st.rerun()
    with col2:
        with st.container(border=True):
            st.subheader("Create an account", text_alignment="left")
            with st.form("register_form"):
                reg_user = st.text_input("Username", key="reg_user", max_chars=64)
                reg_pass = st.text_input(
                    "Password (min 6 characters)", type="password", key="reg_pass",
                    max_chars=128,
                )
                reg_name = st.text_input("Full name (optional)", max_chars=128)
                if st.form_submit_button(
                    "Create account", icon=":material/person_add:"
                ):
                    if len(reg_user) < 3 or len(reg_pass) < 6:
                        st.error("Username must be 3+ chars, password 6+ chars.")
                    else:
                        try:
                            client.register(
                                reg_user, reg_pass, full_name=reg_name or None
                            )
                        except ApiError as exc:
                            st.error(f":material/error: {exc}")
                        else:
                            st.success("Account created — log in to continue.")
    st.stop()

# ---- logged in: navigation -------------------------------------------------

pages = [
    st.Page(
        "app_pages/predict.py",
        title="Analyze a comment",
        icon=":material/chat:",
        default=True,
    ),
    st.Page(
        "app_pages/unseen_test.py",
        title="Test data",
        icon=":material/query_stats:",
    ),
    st.Page(
        "app_pages/chat_agent.py",
        title="Ask the AI agent",
        icon=":material/forum:",
    ),
]
if auth["role"] == "Admin":
    pages.append(
        st.Page(
            "app_pages/feedback.py",
            title="Feedback",
            icon=":material/table_chart:",
        )
    )

page = st.navigation(pages, position="sidebar")
page.run()
