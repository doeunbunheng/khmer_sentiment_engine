"""AI agent - talks with the user about the app's prediction results.

Two modes:
  1. Offline explainer (default, no API key). Deterministic, friendly answers
     built from the actual result the app produced (sentiment, confidence,
     `uncertain` flag, business aspects, emotions, language) or from a dataset
     evaluation report.
  2. LLM upgrade. If AGENT_API_URL + AGENT_API_KEY are set (OpenAI-compatible
     /v1/chat/completions), the question + result context are sent to the
     model; any failure falls back to the offline explainer so chat always
     answers.

Pure logic - no Streamlit import - so it is unit-tested.
"""

import os

import requests

KHMER = {
    "positive": "positive (Vietnamese)",
    "negative": "negative (Vietnamese)",
    "neutral": "neutral (Vietnamese)",
}


def _text(result):
    return str(result.get("text", "") or "")


def _aspects_line(result):
    aspect = (result.get("aspects", {}) or {}).get("business_aspects", {}) or {}
    hits = [name for name, a in aspect.items() if a.get("hit")]
    if not hits:
        return ("No business topic matched (price, service, quality, "
                "authenticity, delivery).")
    keywords = []
    for name in hits:
        keywords.extend(aspect[name].get("keywords", []) or [])
    return ("Business topics found: {}."
            " Matched words: {}.").format(
        ", ".join(hits), ", ".join(keywords[:6]) or "-"
    )


def _emotions_line(result):
    emotions = (result.get("aspects", {}) or {}).get("emotions", {}) or {}
    active = emotions.get("active", []) or []
    if not active:
        return "No emotion fired above the 0.5 threshold."
    scores = emotions.get("scores", {}) or {}
    bits = ", ".join("{} ({:.0%})".format(e, scores.get(e, 0.0)) for e in active)
    return "Active emotions: {}.".format(bits)


def _confidence_line(result):
    conf = float(result.get("confidence", 0.0) or 0.0)
    if result.get("uncertain"):
        return ("Confidence is {:.0%}, below the 90% bar, so this is flagged "
                "`uncertain` - treat it as 'needs review', not a firm guess."
                ).format(conf)
    return ("Confidence is {:.0%} (>= 90%), so the model treats this as a "
            "reliable prediction.").format(conf)


def _sentiment_explain(result, label):
    text = _text(result)
    parts = ["The model classified this comment as {}.".format(label)]
    if text:
        parts.append("Input: {}".format(text))
    parts.append(_confidence_line(result))
    if (result.get("aspects", {}) or {}).get("business_aspects"):
        parts.append(_aspects_line(result))
    if (result.get("aspects", {}) or {}).get("emotions", {}).get("active"):
        parts.append(_emotions_line(result))
    return "\n\n".join(parts)


def _identify(question, result):
    q = (question or "").lower()
    checks = [
        ("positive", ["positive", "posi ", " good"]),
        ("negative", ["negative", " neg"]),
        ("neutral", ["neutral", " neu"]),
        ("uncertain", ["uncertain", "sure", "confiden", "not sure"]),
        ("emotions", ["emotion", "feel", "anger", "angry", "happy", "sad"]),
        ("happened", ["happened", "happening", "what happen", "what happen",
                      "what happen", "whats going", "whats happening",
                      "what did it", "what did this", "why did it show",
                      "show like this", "show like that", "explain this",
                      "explain the result", "tell me what happened",
                      "why did it", "why did this"]),
        ("metrics", ["accuracy", "metric", "score", "dataset", "report",
                     "confusion", "macro", "f1", "per class", "label"]),
        ("aspects", ["aspect", "topic", "about", "price", "service", "quality",
                     "delivery", "authentic", "keywords"]),
        ("language", ["language", "khmer", "english", "mixed"]),
        ("improve", ["improve", "better", "fix", "train", "what should i do",
                     "advice", "next step", "should i", "help me"]),
        ("clear", ["clear", "simple", "short", "plain", "summar"]),
    ]
    for key, words in checks:
        if any(w in q for w in words):
            return key
    return "explain"


def _metrics_explain(result):
    if result.get("type") != "dataset":
        return "This is a single prediction - no dataset metrics to report yet."
    acc = result.get("accuracy")
    f1 = result.get("macro_f1")
    lines = []
    if acc is not None:
        lines.append("Overall accuracy is {:.1%}.".format(acc))
    if f1 is not None:
        lines.append("Macro-F1 is {:.1%}.".format(f1))
    pc = result.get("per_class") or {}
    if pc:
        bits = ", ".join("{} F1 {:.2f}".format(k, v.get("f1", 0)) for k, v in pc.items())
        lines.append("Per class: {}.".format(bits))
    ua = result.get("uncertain_analysis") or {}
    if ua.get("uncertain_rows"):
        lines.append("Rows flagged `uncertain`: {}.".format(ua["uncertain_rows"]))
    dist = result.get("distribution")
    if dist:
        lines.append("Predicted labels: {}.".format(
            ", ".join("{} {}".format(k, v) for k, v in dist.items())))
    return "\n\n".join(lines) or "No dataset metrics available."


def _dist_line(result):
    dist = result.get("distribution") or {}
    return ", ".join("{} {}".format(k, v) for k, v in dist.items()) or "n/a"


def _benchmark_summary(result):
    rows = result.get("rows")
    acc = result.get("accuracy")
    f1 = result.get("macro_f1")
    lines = []
    if acc is not None:
        lines.append(
            "Over {} rows the model hit **{:.2%}** accuracy and **{:.2%}** "
            "macro-F1.".format(rows, acc, f1 or 0.0)
        )
    else:
        lines.append(
            "This run had no true labels, so accuracy cannot be calculated - "
            "it is a prediction overview, not a benchmark. To get accuracy, "
            "check the 'true label' option and prefix lines like "
            "`negative|comment` (or upload a CSV with a label column)."
        )
    pc = result.get("per_class")
    if pc:
        bits = ", ".join("{} F1 {:.2f}".format(k, v.get("f1", 0)) for k, v in pc.items())
        lines.append("Per class: {}".format(bits))
    ua = result.get("uncertain_analysis") or {}
    if ua.get("uncertain_rows"):
        lines.append(
            "{} rows were flagged `uncertain` (confidence < 90%) - review "
            "those before acting on them.".format(ua["uncertain_rows"])
        )
    dist = result.get("distribution")
    if dist:
        lines.append("Predicted labels: {}.".format(_dist_line(result)))
    return "\n\n".join(lines)


def _improve_answer(result):
    return ("To improve this: for every `uncertain` comment (confidence "
            "< 90%), check it yourself and store the correct label - those "
            "become future training data. Keep running the 989-row benchmark "
            "in `Test data` after each retrain so accuracy never regresses.")


def explain(question, result):
    """Offline deterministic answer about `result` for `question`."""
    if not result or not isinstance(result, dict):
        return ("I don't have a result to talk about yet. Run a prediction in "
                "`Analyze a comment` or a test in `Test data` first.")
    intent = _identify(question, result)
    if result.get("type") == "dataset":
        if intent == "improve":
            return _improve_answer(result)
        if intent == "explain":
            return _full_digest(result)
        return _benchmark_summary(result)
    sent = result.get("sentiment", "neutral")
    if intent == "happened":
        return _full_digest(result)
    if intent == "positive":
        return _sentiment_explain(result, "positive")
    if intent == "negative":
        return _sentiment_explain(result, "negative")
    if intent == "neutral":
        return _sentiment_explain(result, "neutral")
    if intent == "uncertain":
        return _confidence_line(result) + "\n\n" + _sentiment_explain(result, sent)
    if intent == "aspects":
        return _aspects_line(result)
    if intent == "emotions":
        return _emotions_line(result)
    if intent == "language":
        lang = result.get("language")
        text = _text(result)
        if lang:
            return ("Detected language: {}.".format(lang) +
                    ("\n\nInput: " + text if text else ""))
        return "Language was not detected for this input."
    if intent == "improve":
        return ("To make the engine more helpful here: collect a human label "
                "for `uncertain` comments and add them to training data; and "
                "regularly run the 989-row benchmark in `Test data` after "
                "every retrain to confirm accuracy does not regress.")
    if intent == "metrics":
        return _explain_metrics(result)
    if intent == "clear":
        lines = [_sentiment_explain(result, result.get("sentiment", "neutral"))]
        if result.get("type") == "dataset":
            lines.append(_explain_metrics(result))
        return "\n\n".join(lines)
    return _full_digest(result)


def _full_digest(result):
    """Complete answer for questions the intent matcher can't place.

    Gives the whole picture of the current result so the agent always
    answers something useful instead of a canned one-liner.
    """
    if result.get("type") == "dataset":
        return _benchmark_summary(result) + "\n\n" + _improve_answer(result)
    sent = result.get("sentiment", "neutral")
    parts = [_sentiment_explain(result, sent)]
    if result.get("language"):
        parts.append("Detected language: {}.".format(result["language"]))
    if (result.get("aspects", {}) or {}).get("business_aspects"):
        parts.append(_aspects_line(result))
    if (result.get("aspects", {}) or {}).get("emotions", {}).get("active"):
        parts.append(_emotions_line(result))
    return "\n\n".join(parts)


def _explain_metrics(result):
    if result.get("type") != "dataset":
        return _full_digest(result)
    return _metrics_explain(result)


def _llm_answer(question, result, url, key, model):
    system = ("You are an assistant inside a Khmer sentiment dashboard. "
              "Explain the result the app produced, clearly and concisely. "
              "Answer in the language the user used.\n\nResult JSON:\n" +
              _result_to_text(result))
    resp = requests.post(
        url + "/chat/completions",
        headers={"Authorization": "Bearer " + key},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["choices"][0]["message"]["content"]


def _result_to_text(result):
    import json
    try:
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(result)


def chat(question, result, url=None, key=None, model=None):
    """Answer a user question about `result`.

    Uses an LLM when configured - via explicit `url`/`key`/`model` (UI
    settings) or the AGENT_API_URL / AGENT_API_KEY / AGENT_MODEL env vars -
    and falls back to the offline explainer on any failure, so the chat
    always answers.
    """
    api_url = (url or os.environ.get("AGENT_API_URL", "")).rstrip("/")
    api_key = key or os.environ.get("AGENT_API_KEY", "")
    api_model = model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")
    if api_url and api_key:
        try:
            return _llm_answer(question, result, api_url, api_key, api_model)
        except Exception:
            pass
    return explain(question, result)