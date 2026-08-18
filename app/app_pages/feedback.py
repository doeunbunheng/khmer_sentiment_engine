"""Feedback (Admin only) — read stored user_feedback rows via GET /feedback."""

import pandas as pd
import streamlit as st

from app.api_client import ApiError
from app.dashboard_utils import current_auth, get_client, handle_api_error

st.title("Feedback")
st.caption(
    "Latest saved predictions (anonymized before storage). Admin role required."
)

auth = current_auth()
if auth is None:
    st.stop()
if auth["role"] != "Admin":
    st.error("Admin role required to view feedback.", icon=":material/lock:")
    st.stop()

client = get_client(st.session_state.dashboard_base_url)

limit = st.select_slider("Rows to load", options=[10, 25, 50, 100, 200, 500], value=50)

if st.button("Load feedback", type="primary", icon=":material/refresh:"):
    try:
        rows = client.feedback(limit=limit, token=auth["token"])
    except ApiError as exc:
        handle_api_error(exc, client, auth)
        st.stop()
    if not rows:
        st.caption("No feedback rows saved yet.")
    else:
        df = pd.DataFrame(rows)
        df["aspects"] = df["aspects"].apply(
            lambda a: ", ".join(
                k for k, v in (a.get("business_aspects", {}) or {}).items() if v.get("hit")
            )
            if isinstance(a, dict)
            else ""
        )
        st.dataframe(df)
        st.caption(f"{len(df)} rows")