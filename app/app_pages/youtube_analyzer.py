"""Analyze YouTube comments — paste a link, collect its comments (free YouTube
Data API), and run the sentiment engine over them through the live API.

New page added on top of the existing dashboard: all original features
(predict, test data, AI agent, feedback) are unchanged.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.dashboard_utils import current_auth, get_client
from app.unseen_eval import LABELS, build_prediction_summary, predict_rows
from src.common.collect import write_collected_csv
from src.collectors.youtube import API_KEY, collect_for_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

st.title("Analyze YouTube comments")
st.caption(
    "Paste any YouTube link, collect its comments for **free** (official "
    "YouTube Data API, no scraping), then run the sentiment engine over them."
)

if not (auth := current_auth()):
    st.stop()
client = get_client(st.session_state.dashboard_base_url)

if not API_KEY:
    st.error(
        "`YOUTUBE_API_KEY` is not set in `.env` — add it (see `.env.example`) "
        "to enable comment collection.",
        icon=":material/key:",
    )
    st.stop()

col_url, col_count = st.columns([3, 1])
with col_url:
    url = st.text_input(
        "YouTube video link",
        placeholder="https://www.youtube.com/watch?v=xxxxxxxxxxx",
    )
with col_count:
    max_comments = st.slider(
        "Max comments", min_value=50, max_value=1000, value=300, step=50
    )

col_a, col_b = st.columns(2)
with col_a:
    run = st.button(
        "Collect & analyze comments",
        type="primary",
        icon=":material/play_arrow:",
    )
with col_b:
    st.caption(
        f"Quota budget: {max_comments * 2 + 5} units\n"
        "(free daily limit is 10,000 units)"
    )

if not run:
    st.info(
        "Paste a YouTube link above, then click **Collect & analyze comments**.",
        icon=":material/youtube_activity:",
    )
    st.stop()

try:
    with st.spinner("Connecting to YouTube API..."):
        video, rows, units = collect_for_url(url, max_comments=max_comments, budget_units=2000)
except ValueError as exc:
    st.error(f":material/error: {exc}")
    st.stop()
except Exception as exc:
    st.error(f":material/error: {exc}")
    st.stop()

if not rows:
    st.warning(
        "This video has no comments (comments may be disabled or the video is "
        "new). Try another video.",
        icon=":material/error:",
    )
    st.stop()

csv_path = RAW_DIR / f"youtube_{video['video_id']}.csv"
saved = write_collected_csv(rows, csv_path)

with st.container(border=True):
    st.markdown(f"**{video['title']}**")
    st.caption(f"Channel: {video['channel']} · {saved} comments · YouTube quota used: {units} units")
    st.download_button(
        "Download collected comments (CSV)",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        icon=":material/download:",
    )

st.caption(f"Saved to `{csv_path}` — also available in the **Test data** page's built-in datasets.")

texts = [r["text"] for r in rows]
progress_bar = st.progress(0.0, text="Sending comments to the sentiment engine...")


def on_progress(done, total):
    progress_bar.progress(done / total, text=f"Predicting {done}/{total} comments...")


preds = predict_rows(
    client, texts, workers=8, progress_cb=on_progress, token=auth["token"]
)
progress_bar.empty()

errors = [p for p in preds if p["sentiment"] is None]
if errors:
    st.error(
        f"{len(errors)} rows failed (usually the API rate limit of 120/min) — "
        f"first: `{errors[0]['error']}`",
        icon=":material/error:",
    )

summary = build_prediction_summary(
    texts,
    [
        (p["sentiment"] if p["sentiment"] is not None else "__error__", p["uncertain"], p["confidence"])
        for p in preds[: len(texts)]
    ],
    aspects=[p.get("aspects") or {} for p in preds[: len(texts)]],
)

with st.container(horizontal=True):
    st.metric("Comments reviewed", summary["rows"], border=True)
    st.metric("Positive", summary["distribution"]["positive"], border=True)
    st.metric("Neutral", summary["distribution"]["neutral"], border=True)
    st.metric("Negative", summary["distribution"]["negative"], border=True)
    st.metric("Uncertain (low confidence)", summary["uncertain_rows"], border=True)

with st.container(border=True):
    st.markdown("**Sentiment distribution**")
    st.bar_chart(
        pd.DataFrame(
            {"count": [summary["distribution"][k] for k in LABELS]},
            index=LABELS,
        )
    )

review = pd.DataFrame(
    [
        {
            "#": p["row"],
            "comment": p["text"],
            "language": p["language"],
            "prediction": p["sentiment"],
            "confidence": p["confidence"],
            "uncertain": p["uncertain"],
        }
        for p in summary["predictions"]
    ]
)
with st.container(border=True):
    st.markdown("**Predictions**")
    st.dataframe(review, use_container_width=True)

report_json = json.dumps(summary, indent=2, ensure_ascii=False)
st.download_button(
    "Download report JSON",
    data=report_json,
    file_name=f"youtube_{video['video_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
    icon=":material/download:",
)

st.session_state["last_eval"] = {
    "type": "dataset",
    "dataset": f"youtube:{video['video_id']}",
    **summary,
}

if st.button(
    "Ask the AI agent about this result",
    type="primary",
    icon=":material/forum:",
):
    st.session_state.chat_starter = (
        "What happened on this dataset? What should I do next?"
    )
    st.switch_page("app_pages/chat_agent.py")