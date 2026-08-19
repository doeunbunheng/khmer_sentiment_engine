"""Test data — benchmark the live API against user-supplied data.

Three ways to bring test data (upload is the default):

1. **Upload a dataset file** — a CSV with a text-ish column (text, comment,
   review, sentence) and optionally a label-ish column (label, sentiment,
   polarity, class). Labels give accuracy metrics; without labels you get a
   prediction-review table.
2. **Paste text** — one comment per line. Optional per-line true label
   (`label|text`, `label: text`, or `label,text`) for accuracy metrics.
3. **Built-in benchmark** — the 989-row kh-polarity held-out set.

Every row is sent to POST /predict with a progress bar (8 threads), then we
show accuracy / macro-F1 / per-class / confusion matrix / by-language plus the
OOD `uncertain` analysis. Reference (offline v2, built-in 989 rows): accuracy
0.8241 · macro-F1 0.7543.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.dashboard_utils import current_auth, get_client
from app.unseen_eval import (
    LABELS,
    build_prediction_summary,
    build_report,
    load_unseen_csv,
    normalize_unseen_df,
    parse_input_text,
    predict_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_EXT_CSV = PROJECT_ROOT / "data" / "external_splits" / "test_ext.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_REPORT = REPORTS_DIR / "dashboard_test_data.json"

st.title("Test data")
st.caption(
    "Send your own comments through the live API: **1. pick a built-in "
    "dataset**, **2. upload a CSV file**, or **3. paste text**."
)

if not (auth := current_auth()):
    st.stop()
client = get_client(st.session_state.dashboard_base_url)

# ---- 1) build the test set ----------------------------------------------------

input_kind = st.segmented_control(
    "Which test input",
    options=["Built-in dataset", "Upload a dataset file", "Paste text"],
    default="Built-in dataset",
    selection_mode="single",
)

source_name = ""
has_labels = False

if input_kind == "Upload a dataset file":
    st.caption(
        "If you have your own dataset, **upload the CSV here** — it needs a "
        f"text column like `text`, `comment` or `review` (e.g. `Comment_Text` "
        f"works) and optionally a label column like `label`, `sentiment` or "
        f"`polarity`. Labels give you accuracy metrics; a file without labels "
        "still shows a prediction-review table."
    )
    uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded is None:
        st.info(
            "Drop your CSV above, or pick **Built-in dataset** / **Paste "
            "text** on the left.",
            icon=":material/upload_file:",
        )
        st.stop()
    source_name = f"uploaded `{uploaded.name}`"
    try:
        df = normalize_unseen_df(pd.read_csv(uploaded, encoding="utf-8"))
    except ValueError as exc:
        st.error(f":material/error: {exc}")
        st.stop()

elif input_kind == "Paste text":
    has_labels = st.checkbox(
        "These lines carry a true label "
        "(format: `negative: bad product`) — compute accuracy.",
        value=False,
    )
    block = st.text_area(
        "Paste comments — one per line",
        height=160,
        placeholder=(
            "សេវាកម្មនៅទីនេះ slow ខ្លាំងណាស់\n"
            "This product is perfect\n"
            "see comment is okay"
        ),
    )
    if not block.strip():
        st.info(
            "Type or paste comments above (one per line).",
            icon=":material/upload:",
        )
        st.stop()
    source_name = "pasted text"
    rows_in = parse_input_text(block, has_labels=has_labels)
    df = pd.DataFrame(rows_in, columns=["text", "label"])

else:
    @st.cache_data(ttl=600)
    def builtin_datasets():
        found = []
        for p in sorted(DATA_DIR.rglob("*.csv")):
            try:
                n = len(pd.read_csv(p, encoding="utf-8"))
            except Exception:
                n = 0
            found.append((str(p.relative_to(DATA_DIR)), str(p), n))
        return found

    datasets = builtin_datasets()
    if not datasets:
        st.error(f"No CSV datasets found under {DATA_DIR}")
        st.stop()
    options = [f"{rel} — {n:,} rows" for rel, _path, n in datasets]
    default_idx = next(
        (i for i, (rel, _p, _n) in enumerate(datasets)
         if rel.endswith("test_ext.csv")),
        0,
    )
    pick = st.selectbox("Built-in dataset", options=options, index=default_idx)
    chosen = dict(zip(options, [p for _r, p, _n in datasets]))[pick]

    @st.cache_data(ttl=600)
    def load_builtin(path):
        return load_unseen_csv(path)

    df = load_builtin(chosen)
    source_name = f"`{pick}`"

n = len(df)
st.caption(f"Dataset: {source_name} · {n} rows")

label_count = int(df["label"].notna().sum()) if "label" in df.columns else 0
if has_labels and label_count == 0:
    st.warning(
        "No lines carried a parseable label — only a prediction review will "
        "be shown (no accuracy).",
        icon=":material/info:",
    )

if n > 100:
    run_options = [
        ("Smoke (50)", 50),
        ("Half", n // 2),
        ("Full", n),
    ]
    labels = [o[0] for o in run_options]
    run_label = st.segmented_control(
        "Rows to send",
        options=labels,
        default="Smoke (50)",
        selection_mode="single",
    )
    max_rows = dict(run_options)[run_label]
else:
    max_rows = n

if max_rows > 120:
    st.info(
        f"This run needs ≈{max_rows} /predict calls — the default API rate "
        "limit is 120/min. Restart the API with "
        "`API_PREDICT_LIMIT=1000/minute` for large runs.",
        icon=":material/info:",
    )

if st.button(
    f"Run the sentiment engine on {max_rows} rows",
    type="primary",
    icon=":material/play_arrow:",
):
    work = df.head(max_rows).copy()
    texts = work["text"].astype(str).tolist()

    progress_bar = st.progress(0.0, text="Sending rows to the API...")

    def on_progress(done, total):
        progress_bar.progress(done / total, text=f"Predicting {done}/{total} rows...")

    preds = predict_rows(
        client, texts, workers=8, progress_cb=on_progress, token=auth["token"]
    )
    progress_bar.empty()

    errors = [p for p in preds if p["sentiment"] is None]
    if errors:
        st.error(
            f"{len(errors)} rows failed (usually the rate limit) — first: "
            f"`{errors[0]['error']}`",
            icon=":material/error:",
        )

    # per-row review table (always shown)
    review = pd.DataFrame(
        [
            {
                "#": i + 1,
                "text": p["text"],
                "language": p.get("language"),
                "true": work["label"].iloc[i] if "label" in work.columns else None,
                "prediction": p["sentiment"] or "error",
                "confidence": p["confidence"],
                "uncertain": p["uncertain"],
            }
            for i, p in enumerate(preds)
            if i < len(work)
        ]
    )
    with st.container(border=True):
        st.markdown("**Predictions**")
        st.dataframe(review)

    labeled_mask = (
        work["label"].notna() & work["label"].ne("")
        if "label" in work.columns
        else pd.Series([False] * len(work))
    )
    labeled_idx = [i for i, okv in enumerate(labeled_mask) if okv]

    if labeled_idx:
        sub_texts = [texts[i] for i in labeled_idx]
        sub_true = [str(work["label"].iloc[i]).lower() for i in labeled_idx]
        sub_results = [
            (
                preds[i]["sentiment"]
                if preds[i]["sentiment"] is not None
                else f"__error__:{preds[i]['error'] or ''}",
                preds[i]["uncertain"],
                preds[i]["confidence"],
                preds[i].get("aspects") or {},
            )
            for i in labeled_idx
        ]
        report = build_report(
            sub_texts,
            sub_true,
            sub_results,
            dataset=source_name or "pasted text",
            endpoint=client.base_url,
        )
        REPORTS_DIR.mkdir(exist_ok=True)
        DEFAULT_REPORT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        st.session_state["last_eval"] = {"type": "dataset", **report}

        with st.container(horizontal=True):
            st.metric(
                "Accuracy",
                f"{report['accuracy']:.1%}",
                delta=f"n={report['rows']}",
                delta_color="off",
                border=True,
            )
            st.metric("Macro-F1", f"{report['macro_f1']:.1%}", border=True)
            st.metric(
                "Uncertain",
                f"{report['uncertain_analysis']['uncertain_rows']}",
                border=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("**Per-class F1**")
                st.bar_chart(
                    pd.DataFrame(
                        {lab: pc["f1"] for lab, pc in report["per_class"].items()},
                        index=["f1"],
                    ).T
                )
        with col2:
            with st.container(border=True):
                st.markdown("**Confusion matrix (rows = true, cols = predicted)**")
                st.dataframe(
                    pd.DataFrame(
                        report["confusion_matrix"]["values"],
                        index=report["confusion_matrix"]["rows"],
                        columns=report["confusion_matrix"]["cols"],
                    )
                )

        with st.container(border=True):
            st.markdown("**OOD guard — uncertain analysis**")
            ua = report["uncertain_analysis"]
            st.caption(
                "Rows flagged `uncertain` (confidence < 0.90) are not "
                "confident enough to present as a hard guess."
            )
            cola, colb, colc, cold = st.columns(4)
            cola.metric("Uncertain rows", ua["uncertain_rows"], border=True)
            colb.metric(
                "Uncertain acc",
                f"{ua['uncertain_accuracy']:.1%}"
                if ua["uncertain_accuracy"] is not None
                else "—",
                border=True,
            )
            colc.metric("Confident rows", ua["confident_rows"], border=True)
            cold.metric(
                "Confident acc",
                f"{ua['confident_accuracy']:.1%}"
                if ua["confident_accuracy"] is not None
                else "—",
                border=True,
            )

        with st.container(border=True):
            st.markdown("**By language**")
            st.dataframe(pd.DataFrame(report["by_language"]).T)

        with st.expander("Cases where it disagreed", icon=":material/error_outline:"):
            if not report["sample_wrong"]:
                st.caption("None.")
            for item in report["sample_wrong"]:
                st.markdown(
                    f"- Row {item['row']}: **true** {item['true']} · **pred** "
                    f"{item['pred']} (conf {item['conf']:.0%}"
                    f"{', uncertain' if item['uncertain'] else ''}) — "
                    f"`{item['text']}`"
                )

        with st.expander("Rows that failed (if any)", icon=":material/error:"):
            if not report["sample_errors"]:
                st.caption("0")
            for item in report["sample_errors"]:
                st.markdown(f"- Row {item['row']}: {item['error']} — `{item['text']}`")

        report_json = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        summary = build_prediction_summary(
            texts,
            [
                (
                    p["sentiment"] if p["sentiment"] is not None else "__error__",
                    p["uncertain"],
                    p["confidence"],
                )
                for p in preds[: len(texts)]
            ],
            aspects=[p.get("aspects") or {} for p in preds[: len(texts)]],
        )
        with st.container(horizontal=True):
            st.metric("Rows reviewed", summary["rows"], border=True)
            st.metric(
                "Uncertain",
                summary["uncertain_rows"],
                border=True,
            )
        st.session_state["last_eval"] = {"type": "dataset", **summary}
        with st.container(border=True):
            st.markdown("**Sentiment distribution**")
            st.bar_chart(pd.DataFrame(
                {"count": [summary["distribution"][k] for k in LABELS]},
                index=LABELS,
            ))
        report_json = json.dumps(summary, indent=2, ensure_ascii=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button(
            "Download report JSON",
            data=report_json,
            file_name=f"dashboard_test_data_{stamp}.json",
            mime="application/json",
            icon=":material/download:",
        )
    with col_dl2:
        st.caption(f"Saved to `{DEFAULT_REPORT}`")

    if st.button(
        "Ask the AI agent about this result",
        type="primary",
        icon=":material/forum:",
    ):
        st.session_state.chat_starter = (
            "What happened on this dataset? What should I do next?"
        )
        st.switch_page("app_pages/chat_agent.py")