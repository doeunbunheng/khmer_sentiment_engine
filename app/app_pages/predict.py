"""Analyze a comment — the main prediction screen.

Sends text to POST /predict and renders the full result: sentiment badge,
confidence, the OOD `uncertain` state (amber panel instead of a guess),
language, business aspect hits + matched keywords, and active emotions.
"""

import streamlit as st

from app.api_client import ApiError
from app.dashboard_utils import current_auth, get_client, handle_api_error

EXAMPLES = {
    "": "",
    "Khmer — positive (product)": (
        "ផលិតផលល្អណាស់ គុណភាពល្អ តម្លៃថោកសមរម្យ"
    ),
    "English — negative (delivery)": (
        "The delivery was late and the price was too high"
    ),
    "Mixed — slow service": (
        "សេវាកម្មនៅទីនេះ slow ខ្លាំងណាស់ មិនចូលចិត្តទាល់តែសោះ"
    ),
    "Neutral — factual question": (
        "តើហាងនេះបើកម៉ោងប៉ុន្មាន?"
    ),
}

SENTIMENT_COLORS = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray",
}
SENTIMENT_KHMER = {
    "positive": "វិជ្ជមាន",
    "negative": "អវិជ្ជមាន",
    "neutral": "អព្យាក្រឹត",
}


def render_result(res):
    sentiment = res["sentiment"]
    confidence = res["confidence"]

    with st.container(horizontal=True):
        st.metric(
            "Sentiment",
            f"{sentiment} · {SENTIMENT_KHMER.get(sentiment, '')}",
            border=True,
        )
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
            delta=None,
            border=True,
        )
        st.metric("Language", res["language"], border=True)

    if res.get("uncertain"):
        st.warning(
            f"The model is **not sure** about this comment "
            f"(confidence {confidence:.0%} < 90%). Don't treat this as a guess — "
            "ask the user or mark it as needs review.",
            icon=":material/warning:",
        )
    else:
        st.success(
            f"Confident prediction ({confidence:.0%} ≥ 90%).",
            icon=":material/verified:",
        )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Business aspects**")
            aspects = res.get("aspects", {}).get("business_aspects", {})
            hits = {name: a for name, a in aspects.items() if a.get("hit")}
            if not hits:
                st.caption("No business aspect matched.")
            else:
                for name, a in hits.items():
                    st.badge(name, icon=":material/check_circle:", color="blue")
                    st.caption(" · ".join(a.get("keywords", [])) or "no keywords")
    with col2:
        with st.container(border=True):
            st.markdown("**Emotions**")
            emotions = res.get("aspects", {}).get("emotions", {})
            active = emotions.get("active", [])
            scores = emotions.get("scores", {})
            if not active:
                st.caption("No emotion above 0.5 threshold.")
            else:
                for e in active:
                    score = scores.get(e, 0.0)
                    st.badge(f"{e} {score:.0%}", icon=":material/sentiment_satisfied:")

    if res.get("saved_id"):
        st.success(
            f"Saved to the database (anonymized, feedback id {res['saved_id']}).",
            icon=":material/save:",
        )
    elif res.get("saved_error"):
        st.caption("Could not save this feedback (consent or storage issue).")


st.title("Analyze a comment")
st.caption(
    "Type or paste a comment in Khmer, English, or mixed — the v2 model "
    "handles all three natively."
)

auth = current_auth()
client = get_client(st.session_state.dashboard_base_url)

example = st.selectbox(
    "Try an example",
    options=list(EXAMPLES),
    key="example_option",
    help=(
        "Loads an example into the box. Your own pasted text is only ever "
        "replaced when you click 'Load example'."
    ),
)
if EXAMPLES[example]:
    if st.button("Load example", key="load_example_btn", icon=":material/play_arrow:"):
        st.session_state["comment_text"] = EXAMPLES[example]
        st.rerun()

with st.form("predict_form"):
    text = st.text_area(
        "Comment",
        height=140,
        max_chars=4000,
        key="comment_text",
        placeholder=(
            "Type or paste a comment here (Khmer, English, or mixed) — up to "
            "4000 characters..."
        ),
    )
    col_consent, col_submit = st.columns([3, 1])
    with col_consent:
        consent = st.checkbox(
            "Save this feedback to the database (anonymized)", value=True
        )
    with col_submit:
        submitted = st.form_submit_button(
            "Analyze", type="primary", icon=":material/analytics:"
        )

if submitted:
    if not text.strip():
        st.error("Enter a comment first.")
    else:
        try:
            res = client.predict(text, consent=consent, token=auth["token"])
        except ApiError as exc:
            handle_api_error(exc, client, auth)
        else:
            st.session_state["last_prediction"] = res
            st.space("small")
            render_result(res)
