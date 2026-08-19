"""Unseen-dataset evaluation through the live API — dashboard version.

Pure logic, no Streamlit import: unit-tested with a fake client in
tests/test_dashboard.py. The Streamlit page (app/app_pages/unseen_test.py)
loads the CSV, runs `run_unseen_eval` with a progress callback, then renders
`build_report` output.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.preprocessing.language_detect import detect_language

LABELS = ["negative", "neutral", "positive"]

TEXT_COLUMNS = ("text", "comment", "review", "sentence", "message", "feedback", "tweet", "statement")
LABEL_COLUMNS = ("label", "sentiment", "polarity", "class", "target", "groundtruth", "truelabel", "gold")


def parse_input_text(block_text, has_labels=False):
    """Parse pasted lines into (text, label_or_None) rows.

    With `has_labels=True`, each line may be `label|text`, `label: text`, or
    `label,text` (case-insensitive). Lines that don't parse a valid label keep
    `label=None`.
    """
    rows = []
    for line in block_text.splitlines():
        line = line.strip()
        if not line:
            continue
        text, label = line, None
        if has_labels:
            for sep in ("|", ":", ","):
                if sep in line:
                    left, _, right = line.partition(sep)
                    if left.strip().lower() in LABELS and right.strip():
                        label = left.strip().lower()
                        text = right.strip()
                        break
        rows.append((text, label))
    return rows


def _row_aspects(p, fallback=None):
    """Aspects/emotions for a row: from the 4th tuple element or a dict.

    Returns {} when nothing meaningful was detected, so the entry simply
    has no `aspects` key.
    """
    if isinstance(p, dict):
        asp = p.get("aspects") or {}
    elif len(p) > 3 and p[3]:
        asp = p[3]
    else:
        asp = fallback or {}
    if not asp:
        return {}
    has_biz = any(
        a.get("hit")
        for a in (asp.get("business_aspects") or {}).values()
    )
    has_emo = bool((asp.get("emotions") or {}).get("active"))
    return asp if (has_biz or has_emo) else {}


def _row_dict(row, text, p, true_label=None, aspects=None):
    """One per-row prediction entry shared by summary and report."""
    entry = {
        "row": row,
        "text": text,
        "language": detect_language(text),
        "sentiment": p[0] if p[0] in LABELS else "neutral",
        "confidence": round(float(p[2] or 0), 3),
        "uncertain": bool(p[1]),
    }
    if true_label is not None:
        entry["true"] = true_label
    asp = _row_aspects(p, aspects)
    if asp:
        entry["aspects"] = asp
    return entry


def build_prediction_summary(texts, results, aspects=None):
    """Prediction-only summary (no true labels): distribution + per-row table."""
    rows_out = []
    dist = {"negative": 0, "neutral": 0, "positive": 0}
    errors = 0
    uncertain = 0
    for i, p in enumerate(results):
        if p is None or p[0].startswith("__error__"):
            errors += 1
            continue
        sent = p[0] if p[0] in LABELS else "neutral"
        dist[sent] += 1
        if p[1]:
            uncertain += 1
        rows_out.append(
            _row_dict(
                i + 1,
                texts[i],
                (p[0], p[1], p[2]),
                aspects=aspects[i] if aspects else None,
            )
        )
    return {
        "rows": len(texts),
        "errors": errors,
        "distribution": dist,
        "uncertain_rows": uncertain,
        "predictions": rows_out,
    }


def _norm_key(col):
    """Normalize a column name for alias matching: lower, strip, drop spaces
    and separators so `Comment_Text`, `comment text`, `COMMENTTEXT` all match."""
    return "".join(
        ch for ch in str(col).strip().lower() if ch not in " _-"
    )


def normalize_unseen_df(df):
    """Map common column aliases, lower-case labels, keep only valid rows.

    Accepts any DataFrame with a text-like column (case-, whitespace- and
    separator-tolerant aliases, e.g. `Comment_Text`); a label-like column is
    optional. Returns a DataFrame with exactly `text` / `label` columns
    (label is None when no label column exists; rows whose label is not in
    LABELS are dropped). Raises ValueError when no text column is found.
    """
    keyed = {_norm_key(c): c for c in df.columns}
    text_col = label_col = None
    for tok in TEXT_COLUMNS:
        if tok in keyed:
            text_col = keyed[tok]
            break
    if text_col is None:
        for key, col in keyed.items():
            if any(tok in key for tok in TEXT_COLUMNS):
                text_col = col
                break
    if text_col is None:
        raise ValueError(
            "no text column found - expected one of: {}; "
            "got columns: {}".format(
                ", ".join(TEXT_COLUMNS), ", ".join(map(str, df.columns))
            )
        )
    for tok in LABEL_COLUMNS:
        if tok in keyed and keyed[tok] != text_col:
            label_col = keyed[tok]
            break
    if label_col is None:
        for key, col in keyed.items():
            if col != text_col and any(tok in key for tok in LABEL_COLUMNS):
                label_col = col
                break
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    if label_col is not None:
        out["label"] = df[label_col].astype(str).str.strip().str.lower()
        out = out[out["label"].isin(LABELS)].reset_index(drop=True)
    else:
        out["label"] = None
    return out


def load_unseen_csv(source):
    """Load a CSV (path or file-like / uploaded bytes), keep valid labels only."""
    df = pd.read_csv(source, encoding="utf-8")
    return normalize_unseen_df(df)


def _fetch(client, text, results, idx, token=None):
    try:
        r = client.predict(text, token=token)
        results[idx] = (
            r.get("sentiment", "neutral"),
            bool(r.get("uncertain", False)),
            r.get("confidence"),
            r.get("aspects") or {},
        )
    except Exception as exc:
        results[idx] = (f"__error__:{exc}", False, None, {})


def run_unseen_eval(client, df, max_rows=None, workers=8, progress_cb=None, token=None):
    """Predict every row through the API (threaded) and return raw results.

    Returns (texts, y_true, results) where results[i] is
    (sentiment, uncertain, confidence); network/API failures are captured as
    (f"__error__:{...}", False, None) so one bad row never kills the run.
    """
    if max_rows:
        df = df.head(max_rows)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].tolist()
    results = [None] * len(texts)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch, client, t, results, i, token): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if progress_cb:
                progress_cb(done, len(texts))
    return texts, y_true, results


def predict_rows(client, texts, workers=8, progress_cb=None, token=None):
    """Run texts through the API (threaded) — no true labels needed.

    Returns a list of dicts aligned with `texts`:
    {text, sentiment, confidence, uncertain, error} where `sentiment` is None
    when the call failed (error holds the message).
    """
    results = [None] * len(texts)

    def _one(idx, text):
        try:
            r = client.predict(text, token=token)
            results[idx] = {
                "text": text,
                "language": detect_language(text),
                "sentiment": r.get("sentiment"),
                "confidence": r.get("confidence"),
                "uncertain": bool(r.get("uncertain", False)),
                "aspects": r.get("aspects") or {},
                "error": None,
            }
        except Exception as exc:
            results[idx] = {
                "text": text,
                "language": detect_language(text),
                "sentiment": None,
                "confidence": None,
                "uncertain": False,
                "aspects": {},
                "error": str(exc)[:120],
            }

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, i, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if progress_cb:
                progress_cb(done, len(texts))
    return results


def build_report(texts, y_true, results, dataset="", endpoint=""):
    """Compute metrics + uncertain analysis from raw eval results."""
    errors = [
        i for i, p in enumerate(results) if p is None or p[0].startswith("__error__")
    ]
    error_set = set(errors)

    y_pred_raw = [p[0] if p and p[0] in LABELS else "neutral" for p in results]
    y_true_idx = [LABELS.index(l) for l in y_true]
    y_pred_idx = [LABELS.index(s) for s in y_pred_raw]

    acc = accuracy_score(y_true_idx, y_pred_idx)
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", labels=[0, 1, 2])
    per_class = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0, 1, 2])

    unc_idx = [i for i, p in enumerate(results) if p and p[1]]
    unc_right = sum(1 for i in unc_idx if y_pred_idx[i] == y_true_idx[i])
    conf_idx = [i for i, p in enumerate(results) if p and not p[1]]
    conf_right = sum(1 for i in conf_idx if y_pred_idx[i] == y_true_idx[i])

    by_lang = {}
    for lang in ("khmer", "english", "mixed"):
        idx = [i for i, t in enumerate(texts) if detect_language(t) == lang]
        if not idx:
            continue
        yt = [y_true_idx[i] for i in idx]
        yp = [y_pred_idx[i] for i in idx]
        by_lang[lang] = {
            "rows": len(idx),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
        }

    sample_errors = [
        {"row": i + 1, "text": texts[i][:120], "error": str(results[i])[:120]}
        for i in errors[:10]
    ]
    predictions = []
    for i in range(len(texts)):
        rp = results[i] or ("neutral", False, None, {})
        predictions.append(
            _row_dict(
                i + 1,
                texts[i],
                (y_pred_raw[i], rp[1], rp[2]),
                true_label=y_true[i],
                aspects=rp[3] if len(rp) > 3 else None,
            )
        )
    sample_wrong = []
    for i in range(len(texts)):
        if len(sample_wrong) >= 10:
            break
        if i in error_set or y_pred_idx[i] == y_true_idx[i]:
            continue
        sample_wrong.append(
            {
                "row": i + 1,
                "text": texts[i][:120],
                "true": y_true[i],
                "pred": y_pred_raw[i],
                "conf": round(float(results[i][2] or 0), 3),
                "uncertain": results[i][1],
            }
        )

    return {
        "endpoint": endpoint,
        "dataset": dataset,
        "rows": len(texts),
        "errors": len(errors),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {
            lab: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for lab, p, r, f, s in zip(LABELS, *per_class)
        },
        "confusion_matrix": {
            "rows": LABELS,
            "cols": LABELS,
            "values": cm.tolist(),
        },
        "by_language": by_lang,
        "uncertain_analysis": {
            "uncertain_rows": len(unc_idx),
            "uncertain_accuracy": round(float(unc_right / len(unc_idx)), 4)
            if unc_idx
            else None,
            "confident_rows": len(conf_idx),
            "confident_accuracy": round(float(conf_right / len(conf_idx)), 4)
            if conf_idx
            else None,
        },
        "sample_errors": sample_errors,
        "sample_wrong": sample_wrong,
        "predictions": predictions,
    }