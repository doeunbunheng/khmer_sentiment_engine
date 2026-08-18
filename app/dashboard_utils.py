"""Shared helpers for dashboard pages (session state, cached API client).

The pages are executed by `st.navigation` after the main script
(app/dashboard.py); this module is the single place where they read the
current auth state and build the API client.
"""

import streamlit as st

from app.api_client import ApiClient


@st.cache_resource
def get_client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


def current_auth():
    """Return the logged-in session {token, user_id, role, username} or None."""
    return st.session_state.get("auth")


def require_auth():
    """Stop the page with a login prompt when there is no session."""
    auth = st.session_state.get("auth")
    if not auth:
        st.warning("Log in from the home screen to continue.", icon=":material/lock:")
        st.stop()
    return auth


def handle_api_error(exc, client, auth):
    """Render a friendly error; clear the session on 401 (expired token)."""
    if getattr(exc, "status", None) == 401:
        st.session_state.auth = None
        st.error("Session expired — log in again.", icon=":material/lock:")
        st.rerun()
    st.error(f":material/error: {exc}")
