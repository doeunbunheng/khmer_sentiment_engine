"""AI agent - talks with the user about the app's prediction results.

Two modes:
  1. Offline explainer (default, no API key). Deterministic, friendly answers
     built from the actual result the app produced - for a single prediction
     (sentiment, confidence, `uncertain` flag, business aspects, emotions,
     language) and for a dataset evaluation report (accuracy, per-class F1,
     confusion matrix, uncertain analysis, by-language, wrong-row samples,
     **per-row predictions**) plus data-driven recommendations.
  2. LLM upgrade. If AGENT_API_URL + AGENT_API_KEY are set (OpenAI-compatible
     /v1/chat/completions), the question + the full active data + conversation
     history are sent to the model; any failure falls back to the offline
     explainer so chat always answers.

The agent is grounded on the WHOLE dataset: summary counts plus every row
(row number, text, sentiment, confidence, `uncertain`, language, aspects,
emotions, true label when present). That lets the user ask:

  - "Why are most comments positive?"        -> distribution + real examples
  - "Show me the uncertain comments"         -> per-row list, lowest first
  - "Why is comment 3 uncertain?"            -> that row, its text + confidence
  - "What are customers complaining about?"  -> negative rows + aspect topics
  - "What happened in this dataset?"         -> friendly overview
  - "Is the model doing well?"               -> accuracy only if labels exist
  - "What should I do next?"                 -> BUSINESS recommendations for
                                              the shop owner (complaints,
                                              delivery, quality, price...),
                                              NOT model-retraining advice

Follow-ups ("why?", "what about this one?", "is it uncertain?") resolve to
the comment that was just discussed via the conversation `history`.

Rules:
- Missing true labels are only mentioned when the question is about
  accuracy/correctness; when the data has no answer the agent says so
  instead of inventing one.
- "What should I do next?"-style questions are read as business questions:
  practical actions based on customer feedback (complaints, repeated
  problems, delivery/quality/price/service). Model-retraining, labeling,
  and accuracy advice is given ONLY when the user asks how to improve the
  model itself.
- A complaint is a signal to CHECK, not proof: for serious or unusual
  complaints the agent recommends verifying the related order/product/
  delivery record before acting.

Pure logic - no Streamlit import - so it is unit-tested.
"""

import os
import re
from collections import Counter

import requests

LABELS = ("negative", "neutral", "positive")


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
        if len(text) > 300:
            parts.append(
                "Input: {}… (long text, {} characters)".format(
                    text[:150].replace("\n", " "), len(text)
                )
            )
        else:
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
        ("most", ["most", "mostly", "majority", "why so many", "so many",
                  "a lot of", "so positive", "so negative", "so neutral"]),
        ("positive", ["positive", "posi ", " good"]),
        ("negative", ["negative", " neg"]),
        ("neutral", ["neutral", " neu"]),
        ("uncertain", ["uncertain", "sure", "confiden", "not sure"]),
        ("complaints", ["complain", "complaint", "dissatisf", "unhappy",
                        "what are customers", "what is bad", "what's bad",
                        "criticism", "problem with", "why are people"]),
        ("doing", ["doing well", "how well", "is the model", "is the ai",
                   "is it good", "reliable", "good job", "performing",
                   "how accurate"]),
        ("emotions", ["emotion", "feel", "anger", "angry", "happy", "sad"]),
        ("confusion", ["confus", "mixed up", "confusing", "matrix",
                       "confused with"]),
        ("wrong", ["wrong", "mistake", "incorrect", "missed", "failed",
                   "fail", "bad at", "get wrong", "went wrong", "not right"]),
        ("weak", ["weak", "worst", "hardest", "struggle", "struggl",
                  "bad class", "problem class", "where is it bad",
                  "least accurate", "weakest"]),
        ("happened", ["happened", "happening", "what happen", "what happen",
                      "what happen", "whats going", "whats happening",
                      "what did it", "what did this", "why did it show",
                      "show like this", "show like that", "explain this",
                      "explain the result", "tell me what happened",
                      "why did it", "why did this"]),
        ("improve_model", ["improve the model", "improve my model",
                           "improve the ai", "make the model", "the model better",
                           "model perform", "retrain", "train the model",
                           "improve accuracy", "model accuracy", "fix the model",
                           "tune the model", "better model", "model itself",
                           "improve the engine"]),
        ("metrics", ["accuracy", "metric", "score", "dataset", "report",
                     "macro", "f1", "per class", "label"]),
        ("aspects", ["aspect", "topic", "about", "price", "service", "quality",
                     "delivery", "authentic", "keywords"]),
        ("language", ["language", "khmer", "english", "mixed"]),
        ("improve", ["improve", "better", "fix", "train", "what should i do",
                     "advice", "next step", "should i", "help me", "suggest",
                     "recommend", "idea", "what do i do"]),
        ("clear", ["clear", "simple", "short", "plain", "summar"]),
    ]
    for key, words in checks:
        if any(w in q for w in words):
            return key
    return "explain"


# ---- per-row prediction helpers -------------------------------------------


def _prediction_rows(result):
    """Per-row prediction dicts in the result, whatever its shape."""
    rows = result.get("predictions")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _find_row(result, n):
    for r in _prediction_rows(result):
        if int(r.get("row") or 0) == int(n):
            return r
    return None


def _rows_with_sentiment(result, label):
    return [r for r in _prediction_rows(result) if r.get("sentiment") == label]


def _uncertain_rows_sorted(result):
    rows = [r for r in _prediction_rows(result) if r.get("uncertain")]
    return sorted(
        rows, key=lambda r: float(r.get("confidence") or 0.0)
    )


def _row_number(text):
    """Extract a row/comment number from free text, or None."""
    t = (text or "").lower()
    for pat in (
        r"comment\s*#?\s*(\d+)",
        r"row\s*#?\s*(\d+)",
        r"(\d+)(?:st|nd|rd|th)\s+comment",
        r"the\s+(\d+)(?:st|nd|rd|th)\b",
        r"comment\s+number\s+(\d+)",
        r"#\s*(\d+)\b",
    ):
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    return None


_EXACT_FOLLOWUPS = {
    "why", "why?", "why is that", "why is it", "why not", "so",
    "what about this one", "what about this", "this one", "and this",
    "is it uncertain", "is it", "really", "hmm", "explain", "explain more",
    "more", "ok", "ok but why", "and why",
}


def _is_followup(question):
    """True for short questions that only make sense with conversation
    context ("why?", "what about this one?", "is it uncertain?")."""
    t = (question or "").strip().lower().rstrip("?.")
    if not t:
        return True
    if t in _EXACT_FOLLOWUPS:
        return True
    return len(t) <= 18 and (
        "why" in t or "what about" in t or "this one" in t or "explain" in t
    )


def _normalize_history(history):
    msgs = []
    for m in history or []:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            role, content = m[0], m[1]
        else:
            continue
        if content:
            msgs.append((str(role or "user"), str(content)))
    return msgs


def _last_mentioned_row(history):
    """Row/comment number the USER most recently discussed in the chat.

    Assistant messages are ignored on purpose: the agent's own answers list
    row numbers ("Comment #3 — ..."), and a question like "why do I have 5
    uncertain comments?" must not be re-routed to one of those rows.
    """
    for role, content in reversed(_normalize_history(history)):
        if role != "user":
            continue
        n = _row_number(content)
        if n is not None:
            return n
    return None


_KHMER_STOP = frozenset("""
ណាស់ ហើយ ដែរ ទេ អី បាន នេះ នោះ ពី ទៅ មក ក្នុង លើ ខ្ញុំ អ្នក
គាត់ គេ យើង មិន អត់ ពេក ខ្លាំង ច្រើន របស់ សម្រាប់ ដូច ដូច្នេះ ក៏
តែ ប៉ុន្តែ ឬ និង ដោយ ទៀត គឺ ជា ដែល ថា មាន ឲ្យ ឱ្យ អោយ ត្រូវ
បាទ ពិត អូ ចាំ បង ចែ ហ្មង ណា ណែ ផង ដែរ ល្អ អីវ៉ាន់ ហើយៗ
""".split())

_EN_STOP = frozenset("""
a an the and or but if then than as at by for from in of on to with about into
over under is are was were be been being do does did have has had can could
will would should may might this that these those it its i you he she we they
them my your our their me us not no so very too just up out all any some more
most much many only one two first also when where why how what which who whom
because while during get got like want just really
""".split())


def _terms_of(text):
    """Content words of a comment (Khmer segmented via khmer-nltk, English
    split), stopwords removed. Used so the agent can talk about whatever
    topic the user's dataset actually covers - not just the fixed business
    aspects."""
    if not text:
        return []
    try:
        from src.preprocessing.segment import segment_text
        words = segment_text(text).lower().split()
    except Exception:
        words = re.findall(r"[a-z]{3,}", (text or "").lower())
    out = []
    for w in words:
        # Khmer words contain combining marks (៌ ្ ំ ...) so isalnum()
        # returns False for them - keep anything with at least one letter.
        if len(w) < 2 or not any(c.isalpha() for c in w):
            continue
        if w in _KHMER_STOP or w in _EN_STOP:
            continue
        out.append(w)
    return out


def _frequent_terms(rows, limit=5):
    """Most repeated content words across rows: [(term, count), ...].

    Terms are counted once per row (a comment repeating a word is not
    double-counted). Singles are kept only when no word repeats, so the
    agent still has something concrete to say for tiny datasets.
    """
    counts = Counter()
    for r in rows:
        counts.update(set(_terms_of(r.get("text"))))
    top = counts.most_common(limit)
    if not top:
        return []
    if top[0][1] > 1:
        top = [(w, c) for w, c in top if c > 1]
    return top


def _row_line(r):
    conf = float(r.get("confidence") or 0.0)
    parts = ["Comment #{} — **{}** — {:.0%}".format(
        r.get("row"), r.get("sentiment"), conf)]
    if r.get("uncertain"):
        parts.append("uncertain")
    text = (r.get("text") or "").strip()
    if text:
        parts.append('"{}"'.format(text[:90]))
    return " ".join(parts)


# ---- dataset analysis helpers ---------------------------------------------


def _confusions(result, top=3):
    """Top true->predicted confusions from the confusion matrix, e.g.
    `neutral` rows predicted `positive`."""
    cm = result.get("confusion_matrix") or {}
    values = cm.get("values") or []
    if not values:
        return []
    rows = cm.get("rows") or LABELS
    cols = cm.get("cols") or LABELS
    pairs = []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if i != j and values[i][j]:
                pairs.append((int(values[i][j]), r, c))
    pairs.sort(reverse=True)
    return pairs[:top]


def _weak_classes(result, threshold=0.75):
    """Per-class F1 sorted weakest-first, e.g. [("neutral", 0.57, 300)]."""
    pc = result.get("per_class") or {}
    items = []
    for lab, d in pc.items():
        if isinstance(d, dict):
            items.append((float(d.get("f1") or 0.0), lab, int(d.get("support") or 0)))
    items.sort()
    return items


def _confusion_answer(result):
    confs = _confusions(result)
    if not confs:
        acc = result.get("accuracy")
        return ("No notable confusion in the matrix - the model keeps the "
                "classes apart cleanly" +
                (" (overall accuracy {:.1%}).".format(acc) if acc is not None
                 else "."))
    lines = ["The biggest confusions in the matrix (true vs predicted):"]
    for n, true_lab, pred_lab in confs:
        lines.append("- **{}** rows got labeled **{}** ({} rows)".format(
            true_lab, pred_lab, n))
    lines.append("That pattern usually means the two classes look alike to "
                 "the model - review those rows and add clearer examples to "
                 "training data.")
    return "\n".join(lines)


def _wrong_answer(result):
    sw = result.get("sample_wrong") or []
    if not sw:
        return _confusion_answer(result)
    lines = ["Here are concrete rows the model got wrong (up to 5):"]
    for it in sw[:5]:
        conf = float(it.get("conf") or 0.0)
        lines.append(
            "- Row {}: **true** {} · **pred** {} (conf {:.0%}{}) — `{}`".format(
                it.get("row"), it.get("true"), it.get("pred"), conf,
                ", uncertain" if it.get("uncertain") else "",
                it.get("text"),
            )
        )
    lines.append("If you see a pattern (e.g. all neutral-ish text or a "
                 "specific topic), that is the class to feed more examples "
                 "of next time.")
    return "\n".join(lines)


def _weak_answer(result):
    weak = _weak_classes(result)
    if not weak:
        return "No per-class F1 breakdown in this result."
    lines = ["Per-class F1, weakest first:"]
    for f1, lab, support in weak:
        lines.append("- {}: F1 **{:.2f}**{}".format(
            lab, f1,
            " (n={})".format(support) if support else ""))
    worst_lab, worst_f1 = weak[0][1], weak[0][0]
    if worst_f1 < 0.75:
        lines.append(
            "`{}` is the weak spot (F1 {:.2f}). It is the hardest class to "
            "recognize - collect more labeled `{}` examples and check the "
            "confusions below.".format(worst_lab, worst_f1, worst_lab))
    else:
        lines.append("No class is badly below the bar - the model is "
                     "balanced across classes.")
    return "\n".join(lines)


def _language_answer(result):
    by = result.get("by_language") or {}
    if not by:
        return "No per-language breakdown in this run."
    lines = ["Accuracy by detected language:"]
    for lang, d in by.items():
        lines.append("- {}: **{:.1%}** (n={}, macro-F1 {:.1%})".format(
            lang, float(d.get("accuracy") or 0.0),
            d.get("rows") or 0, float(d.get("macro_f1") or 0.0)))
    worst = min(by.items(), key=lambda kv: float(kv[1].get("accuracy") or 0.0))
    if worst[1].get("accuracy") is not None and worst[1]["accuracy"] < 0.75:
        lines.append("`{}` is the weakest - add more {} rows to training "
                     "data.".format(worst[0], worst[0]))
    return "\n".join(lines)


_ASPECT_ACTIONS = {
    "Delivery": "Check delayed orders and whether customers receive updates "
                "while waiting.",
    "Product Quality": "Check the products customers said were poor quality "
                       "or different from the photos.",
    "Price": "Check whether the price matches the quality customers receive.",
    "Service": "Review how quickly your page responds when customers contact "
               "you.",
    "Authenticity": "Review the product's supplier and origin, and make sure "
                    "the listing describes it accurately.",
}

_WRONG_ORDER_WORDS = ("ពណ៌", "color", "ខុស", "wrong", "មិនដូច",
                      "not like", "different from", "បែរជា", "instead")


def _business_recommendations(result):
    """'What should I do next?' answered for the shop owner, not the ML
    engineer: practical actions grounded on the actual complaints.

    Technical advice (retrain, labeling, accuracy) is deliberately NOT
    included here - it lives in `_recommendations`, which is only reached
    when the user asks how to improve the model itself.
    """
    neg = _rows_with_sentiment(result, "negative")
    commerce = _is_commerce_data(neg)
    topics = Counter()
    if commerce:
        for r in neg:
            for name, a in ((r.get("aspects") or {}).get("business_aspects")
                            or {}).items():
                if a.get("hit"):
                    topics[name] += 1
    lines = []
    if neg:
        lines.append(
            "Based on the customer feedback, I recommend checking the "
            "problems customers mentioned first - **{} comments were "
            "negative**.".format(len(neg))
        )
    else:
        lines.append(
            "Based on the customer feedback, there were no negative "
            "comments - the ones to keep an eye on are the neutral "
            "comments and the 'not sure' cases below."
        )
    if topics:
        names = ", ".join("**{}**".format(n) for n, _ in topics.most_common())
        lines.append("The main complaints are about: {}.".format(names))
        lines.append("A good next step would be:")
        for name, count in topics.most_common():
            lines.append(
                "- **{}** (mentioned {}×) — {}".format(
                    name, count, _ASPECT_ACTIONS.get(
                        name, "Check what the customers are describing."
                    )
                )
            )
    else:
        terms = _frequent_terms(neg) if neg else []
        if terms:
            bits = ", ".join(
                "**{}** ({}×)".format(w, c) for w, c in terms
            )
            lines.append(
                "The words customers mention most are: {}.".format(bits)
            )
            lines.append("A good next step would be:")
            for w, _c in terms:
                lines.append(
                    "- **{}** — Look into what people describe when they "
                    "mention this, and check what is really going on "
                    "behind it.".format(w)
                )
        elif not neg:
            lines.append(
                "Keep replying to the neutral comments too - they may "
                "become complaints or praise with a quick follow-up."
            )
    wrong = []
    if commerce:
        wrong = [
            r for r in neg
            if any(w in (r.get("text") or "") for w in _WRONG_ORDER_WORDS)
        ]
    if wrong:
        lines.append(
            "- **Wrong orders** — some customers said they received the "
            "wrong item or color; check the packing and shipping process "
            "for those orders."
        )
    if len(neg) >= 2:
        if commerce:
            lines.append(
                "Start with the negative comments first, especially "
                "repeated complaints - if several people report the same "
                "issue, it may be a real problem with that product, "
                "delivery, or service."
            )
        else:
            lines.append(
                "Start with the negative comments first, especially "
                "repeated complaints - if several people describe the same "
                "issue, it is probably a real problem worth investigating."
            )
    ua = result.get("uncertain_analysis") or {}
    unc = ua.get("uncertain_rows")
    if unc is None:
        unc = result.get("uncertain_rows")
    if unc:
        lines.append(
            "{} comments were flagged as 'not sure' (the system is not "
            "confident about them) - read those by hand before deciding, "
            "since the prediction may not be reliable.".format(unc)
        )
    if commerce:
        lines.append(
            "Before acting, verify the related order, product, delivery, or "
            "customer-service record - a complaint is a signal to check, not "
            "proof that it happened."
        )
    else:
        lines.append(
            "Before acting, verify the details behind each complaint - a "
            "comment is a signal to check, not proof that it happened."
        )
    return "\n\n".join(lines)


def _recommendations(result):
    """Data-driven 'what should I do next' built from the actual report."""
    rec = []
    ua = result.get("uncertain_analysis") or {}
    unc = ua.get("uncertain_rows")
    if unc is None:
        unc = result.get("uncertain_rows")
    if unc:
        rec.append(
            "**{} rows were flagged `uncertain`** (confidence < 90%) - "
            "review those first and store your corrected label; they become "
            "future training data.".format(unc)
        )
    weak = _weak_classes(result)
    if weak and weak[0][0] < 0.75:
        rec.append("**`{}` is the weakest class (F1 {:.2f})** - collect more "
                   "labeled `{}` examples, especially the confusing ones "
                   "shown above.".format(weak[0][1], weak[0][0], weak[0][1]))
    by = result.get("by_language") or {}
    if by:
        worst_lang, worst_d = min(
            by.items(), key=lambda kv: float(kv[1].get("accuracy") or 0.0)
        )
        if worst_d.get("accuracy") is not None and worst_d["accuracy"] < 0.75:
            rec.append("**{} rows are the weakest ({:.1%} acc)** - add more "
                       "{} training data.".format(
                worst_lang, worst_d["accuracy"], worst_lang))
    if result.get("errors"):
        rec.append("**{} rows failed** (rate limit / network) - re-run them "
                   "after raising `API_PREDICT_LIMIT`.".format(result["errors"]))
    if result.get("accuracy") is None:
        rec.append("**Add true labels** to the dataset (a label/sentiment/"
                   "polarity column) and rerun to get accuracy metrics.")
    if not rec:
        rec.append("**The result looks healthy** - accuracy and per-class "
                   "balance are good. Keep this dataset as a regression "
                   "benchmark.")
    if unc is None or not unc:
        rec.append("After any change (retrain, new keywords, threshold), "
                   "rerun this benchmark and watch the `uncertain` rate and "
                   "accuracy - they must not regress.")
    return ("To improve this:\n\n" +
            "\n\n".join("- " + r for r in rec))


def _metrics_explain(result):
    if result.get("type") != "dataset":
        return "This is a single prediction - no dataset metrics to report yet."
    acc = result.get("accuracy")
    f1 = result.get("macro_f1")
    lines = []
    if acc is not None:
        lines.append("Overall accuracy is {:.1%}.".format(acc))
    else:
        lines.append("This run had no true labels, so accuracy cannot be "
                     "calculated - it is a prediction overview, not a "
                     "benchmark.")
    if f1 is not None:
        lines.append("Macro-F1 is {:.1%}.".format(f1))
    pc = result.get("per_class") or {}
    if pc:
        bits = ", ".join("{} F1 {:.2f}".format(k, v.get("f1", 0))
                         for k, v in pc.items() if isinstance(v, dict))
        lines.append("Per class: {}.".format(bits))
    ua = result.get("uncertain_analysis") or {}
    if ua.get("uncertain_rows"):
        lines.append("Rows flagged `uncertain`: {}.".format(ua["uncertain_rows"]))
    dist = result.get("distribution")
    if dist:
        lines.append("Predicted labels: {}.".format(
            ", ".join("{} {}".format(k, v) for k, v in dist.items())))
    return "\n\n".join(lines) or "No dataset metrics available."


def _short_dataset_summary(result):
    """Short answer for 'clear/simple/summarize' questions on a dataset."""
    acc = result.get("accuracy")
    head = ("This run scored **{:.1%}** accuracy (macro-F1 **{:.1%}**) over "
            "{} rows.").format(acc, result.get("macro_f1") or 0.0, result.get("rows")) \
        if acc is not None else \
        "This run reviewed {} rows with no true labels (prediction overview).".format(
            result.get("rows"))
    ua = result.get("uncertain_analysis") or {}
    unc = ua.get("uncertain_rows")
    if unc is None:
        unc = result.get("uncertain_rows")
    tail = ""
    if unc:
        tail = " {} rows were `uncertain`.".format(unc)
    return head + tail


def _happened_answer(result):
    """Friendly 'what happened in this dataset' overview."""
    rows = result.get("rows")
    dist = result.get("distribution") or {}
    present = {k: v for k, v in dist.items() if v}
    acc = result.get("accuracy")
    ua = result.get("uncertain_analysis") or {}
    unc = ua.get("uncertain_rows")
    if unc is None:
        unc = result.get("uncertain_rows")
    errors = result.get("errors") or 0
    lines = ["You analyzed **{}** comments.".format(rows)]
    if present:
        total = sum(present.values()) or rows or 0
        ordered = sorted(present.items(), key=lambda kv: -kv[1])
        top, top_n = ordered[0]
        rest = ", ".join(
            "{} {}".format(v, k) for k, v in ordered[1:] if v
        ) or "none"
        lines.append(
            "Most were predicted **{}** ({} of {}), while a smaller number "
            "were {}.".format(top, top_n, total, rest)
        )
    if acc is not None:
        lines.append(
            "On the labeled rows the model reached **{:.1%}** accuracy "
            "(macro-F1 **{:.1%}**).".format(
                acc, result.get("macro_f1") or 0.0
            )
        )
    if unc:
        lines.append(
            "**{} predictions were uncertain** (confidence below 90%) - "
            "those comments are the best ones to review by hand.".format(unc)
        )
    elif unc == 0 and present:
        lines.append("No predictions were uncertain - the model was "
                     "confident on every row.")
    if errors:
        lines.append("{} rows failed during the run and are not "
                     "included.".format(errors))
    if present:
        top_label = sorted(present.items(), key=lambda kv: -kv[1])[0][0]
        lines.append(
            "Overall the data suggests mostly {} customer sentiment, "
            "but the {} rows deserve attention.".format(
                "favorable" if top_label == "positive" else top_label,
                "complaint" if top_label != "positive" else "neutral/negative",
            )
        )
    return "\n\n".join(lines)


_CLASS_READING = {
    "positive": "The model reads these as generally favorable - praise, "
                "satisfaction, or approval in the wording.",
    "negative": "The model reads these as complaints, problems, or "
                "dissatisfaction.",
    "neutral": "The model reads these as factual, mixed, or non-opinionated "
               "wording - neither clearly positive nor negative.",
}


def _most_answer(result):
    """'Why are most comments positive/negative/neutral?'"""
    dist = result.get("distribution") or {}
    present = {k: v for k, v in dist.items() if v}
    if not present:
        return ("No sentiment distribution is available in this result, so "
                "I can't say which class dominates.")
    ordered = sorted(present.items(), key=lambda kv: -kv[1])
    top, top_n = ordered[0]
    total = sum(present.values()) or result.get("rows") or 0
    lines = [
        "Most comments were predicted **{}**: {} of {} comments "
        "(about {:.0%}).".format(
            top, top_n, total, top_n / total if total else 0.0
        )
    ]
    rest = ", ".join("{} {}".format(v, k) for k, v in ordered[1:] if v)
    lines.append(
        "There are still {} - so the feedback is not completely "
        "one-sided.".format(rest or "none of the other classes")
    )
    lines.append(_CLASS_READING.get(top, ""))
    examples = _rows_with_sentiment(result, top)[:4]
    if examples:
        lines.append("For example:")
        lines.extend("- " + _row_line(r) for r in examples)
    return "\n\n".join(lines)


_COMMERCE_MARKERS = {
    # Khmer - words that ONLY make sense when buying/selling/delivering
    # goods. Ambiguous words like តម្លៃ (value), ថ្លៃ (expensive/fee),
    # គុណភាព (quality), ពិត (real) are deliberately NOT here - they
    # appear in any domain (education, politics, health...).
    "ដឹកជញ្ជូន", "អូឌ័រ", "កម្ម៉ង់", "ផលិតផល", "អីវ៉ាន់",
    "ទំនិញ", "ហាង", "លក់", "ទិញ", "អាវ", "ខោ", "ស្បែកជើង",
    "អតិថិជន", "ម៉ូយ", "កាត់តម្លៃ", "បញ្ចុះតម្លៃ",
}
_EN_COMMERCE = (
    "delivery", "shipping", "deliver", "parcel", "package", "product",
    "shop", "store", "buy", "sell", "purchase", "item", "shirt",
    "courier", "seller", "vendor", "customer", "order number",
)
_EN_COMMERCE_RE = re.compile(
    r"\b(" + "|".join(_EN_COMMERCE) + r")\b", re.IGNORECASE
)


def _is_commerce_data(rows):
    """True only when the comments clearly talk about buying / selling /
    delivering goods. E-commerce advice (suppliers, packing, orders) is
    given ONLY for such datasets - otherwise the agent talks about the
    dataset's own words instead."""
    for r in rows:
        text = r.get("text") or ""
        if _EN_COMMERCE_RE.search(text):
            return True
        low = text.lower()
        if any(m in low for m in _COMMERCE_MARKERS):
            return True
    return False


def _class_one_sentence(rows, label):
    """'In one sentence' summary of a class - the topics/aspects the rows
    actually mention, so the user does not have to read every comment."""
    n = len(rows)
    if not n:
        return ""
    commerce = _is_commerce_data(rows)
    topics = Counter()
    if commerce:
        for r in rows:
            for name, a in ((r.get("aspects") or {}).get("business_aspects")
                            or {}).items():
                if a.get("hit"):
                    topics[name] += 1
    terms = None
    if not topics:
        t = _frequent_terms(rows, limit=3)
        if t:
            terms = ", ".join("**{}**".format(w) for w, _ in t)
    if label == "negative":
        head = ("In one sentence: these {} negative comments are mostly "
                "about ".format(n))
        if topics:
            bits = ", ".join("**{}**".format(t) for t, _ in topics.most_common())
            return head + bits + "."
        if terms:
            return head + terms + "."
        return ("In one sentence: these {} comments express clear "
                "dissatisfaction, but no single topic repeats enough to "
                "name one.".format(n))
    if label == "positive":
        if topics:
            bits = ", ".join("**{}**".format(t) for t, _ in topics.most_common())
            return ("In one sentence: these {} positive comments are mostly "
                    "happy about {}.".format(n, bits))
        if terms:
            return ("In one sentence: these {} positive comments often "
                    "mention {}.".format(n, terms))
        return ("In one sentence: these {} comments read as praise, "
                "satisfaction, or approval in the wording.".format(n))
    if topics:
        bits = ", ".join("**{}**".format(t) for t, _ in topics.most_common())
        return ("In one sentence: these {} neutral comments mostly touch "
                "{} without strong feeling.".format(n, bits))
    return ("In one sentence: these {} neutral comments neither praise nor "
            "attack - mostly factual or balanced wording.".format(n))


_CLASS_ACTION = (
    "What to do: check the details behind these comments, reply to each "
    "person, and fix the most repeated problem first - if several people "
    "mention the same issue, it is usually a real problem worth "
    "investigating."
)


def _class_answer(result, label):
    """'Why are the comments positive/negative/neutral?' (no 'most')."""
    dist = result.get("distribution") or {}
    n = dist.get(label, 0)
    total = sum(v for v in dist.values()) or result.get("rows") or 0
    lines = [
        "**{}** of {} comments were predicted **{}** (about {:.0%} of the "
        "run).".format(n, total, label, n / total if total else 0.0)
    ]
    lines.append(_CLASS_READING.get(label, ""))
    examples = _rows_with_sentiment(result, label)
    if examples:
        lines.append(_class_one_sentence(examples, label))
        lines.append("For example:")
        lines.extend("- " + _row_line(r) for r in examples[:3])
        if len(examples) > 3:
            lines.append(
                "- ... and {} more (ask \"show me all {} comments\" for the "
                "full list)".format(len(examples) - 3, label)
            )
    if label == "negative" and len(examples) >= 2:
        lines.append(_CLASS_ACTION)
    return "\n\n".join(lines)


def _class_list_answer(result, label):
    """'Show me all the negative/positive/neutral comments.' - full list."""
    rows = _rows_with_sentiment(result, label)
    if not rows:
        return ("No comments were predicted **{}** in this result.".format(
            label
        ))
    lines = [
        "There are {} comments predicted **{}**:".format(len(rows), label)
    ]
    lines.extend("- " + _row_line(r) for r in rows)
    return "\n\n".join(lines)


def _uncertain_explain_answer(result):
    """'Why do I have N uncertain comments?'"""
    rows_unc = _uncertain_rows_sorted(result)
    n = len(rows_unc)
    if not n:
        return ("No comments were flagged `uncertain` - every prediction "
                "met the 90% confidence bar.")
    lines = [
        "{} comments were flagged `uncertain` because their confidence fell "
        "below the 90% bar - the model was not sure enough to present a "
        "firm guess.".format(n)
    ]
    lines.append(
        "That usually happens with unclear wording, short or mixed-language "
        "text, slang, or comments that mix positive and negative signals."
    )
    ua = result.get("uncertain_analysis") or {}
    if ua.get("uncertain_accuracy") is not None:
        lines.append(
            "On the labeled rows, only {:.0%} of the uncertain predictions "
            "were correct - reviewing them is exactly the right move.".format(
                ua["uncertain_accuracy"]
            )
        )
    lines.append("Lowest confidence first:")
    lines.extend("- " + _row_line(r) for r in rows_unc[:5])
    if n > 5:
        lines.append("- ... and {} more (ask \"show me the uncertain "
                     "comments\" for the full list)".format(n - 5))
    return "\n\n".join(lines)


def _uncertain_list_answer(result):
    """'Show me the uncertain comments.'"""
    rows_unc = _uncertain_rows_sorted(result)
    if not rows_unc:
        return ("There are no uncertain comments in this result - every "
                "prediction met the 90% confidence bar.")
    lines = [
        "There are {} uncertain comments (confidence below 90%):".format(
            len(rows_unc)
        )
    ]
    lines.extend("- " + _row_line(r) for r in rows_unc)
    lines.append(
        'Ask about any of them, e.g. "why is comment #{} uncertain?".'.format(
            rows_unc[0]["row"]
        )
    )
    return "\n\n".join(lines)


def _row_answer(result, n):
    """Explain a specific comment/row by its number."""
    row = _find_row(result, n)
    if row is None:
        rows = result.get("rows")
        return (
            "There is no comment #{} in this result{}.".format(
                n, " (it has {} rows)".format(rows) if rows else ""
            )
        )
    text = (row.get("text") or "").strip()
    sent = row.get("sentiment", "neutral")
    conf = float(row.get("confidence") or 0.0)
    unc = bool(row.get("uncertain"))
    lines = []
    if text:
        lines.append("Comment #{} says: \"{}\"".format(n, text))
    lines.append(
        "The model predicted **{}** with {:.0%} confidence.".format(
            sent, conf
        )
    )
    if unc:
        lines.append(
            "That is below the 90% bar, so it is flagged `uncertain` - the "
            "wording is probably unclear, mixed, or short, and the model "
            "could not commit to a firm guess."
        )
    else:
        lines.append(
            "That meets the 90% confidence bar, so it is treated as a "
            "reliable prediction."
        )
    asp = row.get("aspects") or {}
    hits = [
        name for name, a in (asp.get("business_aspects") or {}).items()
        if a.get("hit")
    ]
    if hits:
        lines.append("Business topics matched: {}.".format(", ".join(hits)))
    active = (asp.get("emotions") or {}).get("active") or []
    if active:
        lines.append("Active emotions: {}.".format(", ".join(active)))
    if row.get("true") is not None:
        ok = str(row["true"]) == str(sent)
        lines.append(
            "The true label is **{}** - the prediction was {}.".format(
                row["true"], "correct" if ok else "wrong"
            )
        )
    return "\n\n".join(lines)


def _complaints_answer(result):
    """'What are customers complaining about?' - works with the fixed
    business aspects when they hit, and falls back to the most repeated
    words in the negative comments when the dataset is about something
    else entirely (restaurants, services, politics...)."""
    neg = _rows_with_sentiment(result, "negative")
    if not neg:
        return ("No comments were predicted **negative** in this result, so "
                "there are no complaints to summarize.")
    commerce = _is_commerce_data(neg)
    topics = Counter()
    if commerce:
        for r in neg:
            for name, a in ((r.get("aspects") or {}).get("business_aspects")
                            or {}).items():
                if a.get("hit"):
                    topics[name] += 1
    lines = [
        "The **{} comments predicted negative** - here is what the data "
        "shows:".format(len(neg))
    ]
    if topics:
        bits = ", ".join(
            "**{}** ({}×)".format(name, count)
            for name, count in topics.most_common()
        )
        lines.append("The main complaint topics are: {}.".format(bits))
    else:
        terms = _frequent_terms(neg)
        if terms:
            bits = ", ".join(
                "**{}** ({}×)".format(w, c) for w, c in terms
            )
            lines.append(
                "The words customers mention most in these comments: "
                "{}.".format(bits)
            )
        else:
            lines.append(
                "The comments express dissatisfaction, but no specific "
                "topic repeats often enough to name one."
            )
    lines.append("For example:")
    lines.extend("- " + _row_line(r) for r in neg[:4])
    if len(neg) > 4:
        lines.append("- ... and {} more negative comments.".format(len(neg) - 4))
    lines.append(
        "These rows are the ones to act on first - fix the issue they "
        "mention and re-run to see the negative rate drop."
    )
    return "\n\n".join(lines)


def _doing_answer(result):
    """'Is the model doing well?' - accuracy-based, labels-aware."""
    acc = result.get("accuracy")
    if acc is None:
        return (
            "This dataset does not contain true labels, so accuracy cannot "
            "be measured from this run. However, we can still inspect the "
            "prediction distribution, confidence scores, and uncertain "
            "comments - try \"what happened?\" or \"show me the uncertain "
            "comments\"."
        )
    lines = [
        "On this dataset the model reached **{:.1%}** accuracy and "
        "**{:.1%}** macro-F1.".format(acc, result.get("macro_f1") or 0.0)
    ]
    if acc >= 0.90:
        verdict = "very strong"
    elif acc >= 0.80:
        verdict = "solid"
    elif acc >= 0.70:
        verdict = "decent, but with clear room to improve"
    else:
        verdict = "below what we would like"
    lines.append(
        "That is {} for this kind of text.".format(verdict)
    )
    weak = _weak_classes(result)
    if weak:
        bits = ", ".join(
            "{} F1 {:.2f}".format(lab, f1) for f1, lab, _s in weak
        )
        lines.append("Per class: {}.".format(bits))
        if weak[0][0] < 0.75:
            lines.append(
                "`{}` predictions are the weakest (F1 {:.2f}) - that class "
                "is the one to feed more labeled examples of.".format(
                    weak[0][1], weak[0][0]
                )
            )
    ua = result.get("uncertain_analysis") or {}
    if ua.get("uncertain_rows"):
        lines.append(
            "{} rows were `uncertain` (confidence < 90%); their accuracy "
            "was {:.0%}, so double-check those before acting on "
            "them.".format(
                ua["uncertain_rows"], ua.get("uncertain_accuracy") or 0.0
            )
        )
    return "\n\n".join(lines)


def _wants_advice(q):
    return any(w in q for w in (
        "suggest", "recommend", "idea", "advice", "should i do",
        "what do i do", "help me", "next step",
    ))


def _single_advice(result):
    """Practical next step for ONE comment (negative / positive / neutral)."""
    sent = result.get("sentiment", "neutral")
    topics = [
        name for name, a in ((result.get("aspects") or {}).get("business_aspects")
                             or {}).items() if a.get("hit")
    ]
    topic_bits = (", ".join(topics) + " - ") if topics else ""
    if sent == "negative":
        return (
            "Suggested idea: this is a dissatisfied customer. Verify the "
            "related {}order / delivery / customer-service record first, "
            "reply politely to the customer, and fix the issue mentioned "
            "before asking for a review.".format(topic_bits)
        )
    if sent == "positive":
        return (
            "Suggested idea: this customer is happy - reply briefly to "
            "thank them and invite them to buy again or leave a review."
        )
    return (
        "Suggested idea: neutral comments are 'not sure yet' - ask a quick "
        "follow-up question or check the order record to see if anything "
        "was off."
    )


def explain(question, result, history=None):
    """Offline deterministic answer about `result` for `question`.

    `history` is an optional list of prior {role, content} messages (or
    (role, content) tuples) so short follow-ups ("why?", "what about this
    one?") resolve to the comment that was just discussed.
    """
    if not result or not isinstance(result, dict):
        return ("I don't have a result to talk about yet. Run a prediction in "
                "`Analyze a comment` or a test in `Test data` first.")
    q = (question or "").lower()
    intent = _identify(question, result)
    if result.get("type") == "dataset":
        row_num = _row_number(question)
        if row_num is None and _is_followup(q):
            row_num = _last_mentioned_row(history)
        if row_num is not None:
            return _row_answer(result, row_num)
        if intent == "most":
            return _most_answer(result)
        if intent in ("positive", "negative", "neutral"):
            if any(w in q for w in ("show", "list", "all", "which", "see",
                                    "every")):
                return _class_list_answer(result, intent)
            answer = _class_answer(result, intent)
            if _wants_advice(q):
                answer += "\n\n" + _business_recommendations(result)
            return answer
        if intent == "uncertain":
            if any(w in q for w in ("show", "list", "which", "see")):
                return _uncertain_list_answer(result)
            return _uncertain_explain_answer(result)
        if intent == "complaints":
            return _complaints_answer(result)
        if intent == "doing":
            return _doing_answer(result)
        if intent == "happened":
            return _happened_answer(result)
        if intent == "confusion":
            return _confusion_answer(result)
        if intent == "wrong":
            return _wrong_answer(result)
        if intent == "weak":
            return _weak_answer(result)
        if intent == "improve_model":
            return _recommendations(result)
        if intent == "improve":
            return _business_recommendations(result)
        if intent == "metrics":
            return _metrics_explain(result)
        if intent == "language":
            return _language_answer(result)
        if intent == "clear":
            return _short_dataset_summary(result)
        return _full_digest(result)
    sent = result.get("sentiment", "neutral")
    if intent == "happened":
        return _full_digest(result)
    if intent in ("positive", "negative", "neutral"):
        answer = _sentiment_explain(result, intent)
        if _wants_advice(q):
            answer += "\n\n" + _single_advice(result)
        return answer
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
        parts = [_happened_answer(result)]
        weak = _weak_classes(result)
        if weak and weak[0][0] < 0.75:
            parts.append(_weak_answer(result))
        confs = _confusions(result)
        if confs:
            parts.append(_confusion_answer(result))
        parts.append(_business_recommendations(result))
        return "\n\n".join(parts)
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


def _active_data_text(result):
    """Compact ACTIVE DATA block for the LLM: summary + per-row predictions
    (capped so large datasets stay inside the model's context)."""
    if result.get("type") != "dataset":
        return _result_to_text(result)
    lines = [
        "dataset: {}\nrows: {}\nerrors: {}".format(
            result.get("dataset") or "unknown",
            result.get("rows"),
            result.get("errors") or 0,
        )
    ]
    dist = result.get("distribution") or {}
    if dist:
        lines.append("distribution: {}".format(
            ", ".join("{}={}".format(k, v) for k, v in dist.items())
        ))
    acc = result.get("accuracy")
    if acc is not None:
        lines.append(
            "accuracy: {:.4f}\nmacro_f1: {:.4f}".format(
                acc, result.get("macro_f1") or 0.0
            )
        )
    ua = result.get("uncertain_analysis") or {}
    if ua.get("uncertain_rows") is not None:
        lines.append(
            "uncertain_rows: {} (confidence < 0.90)".format(
                ua["uncertain_rows"]
            )
        )
    if result.get("per_class"):
        lines.append("per_class_f1: {}".format(
            ", ".join(
                "{}={}".format(k, v.get("f1"))
                for k, v in result["per_class"].items()
                if isinstance(v, dict)
            )
        ))
    rows = _prediction_rows(result)
    cap = 60
    lines.append("predictions ({} rows{}):".format(
        len(rows),
        ", first {} shown".format(cap) if len(rows) > cap else "",
    ))
    for r in rows[:cap]:
        bits = [
            "row {}".format(r.get("row")),
            "sentiment {}".format(r.get("sentiment")),
            "conf {:.3f}".format(float(r.get("confidence") or 0.0)),
        ]
        if r.get("uncertain"):
            bits.append("uncertain")
        if r.get("true") is not None:
            bits.append("true {}".format(r["true"]))
        bits.append("lang {}".format(r.get("language") or "?"))
        text = (r.get("text") or "").replace("\n", " ")[:120]
        bits.append("text '{}'".format(text))
        asp = r.get("aspects") or {}
        hits = [
            name for name, a in (asp.get("business_aspects") or {}).items()
            if a.get("hit")
        ]
        if hits:
            bits.append("aspects {}".format(",".join(hits)))
        active = (asp.get("emotions") or {}).get("active") or []
        if active:
            bits.append("emotions {}".format(",".join(active)))
        lines.append(" - ".join(bits))
    if len(rows) > cap:
        lines.append("... {} more rows not shown (use the offline explainer "
                     "for row-specific questions)".format(len(rows) - cap))
    return "\n".join(lines)


def _llm_answer(question, result, url, key, model, history=None):
    system = (
        "You are an assistant that discusses the user's analyzed sentiment "
        "dataset. The ACTIVE DATA below is the prediction result the "
        "system produced - use ONLY this data, never invent rows, numbers, "
        "texts, or metrics.\n\n"
        "Rules:\n"
        "- Answer in the language the user used.\n"
        "- NEVER copy or repeat the ACTIVE DATA or the conversation - "
        "answer directly and concisely (2-6 sentences).\n"
        "- Use the CONVERSATION for follow-ups: 'why?', 'what about this "
        "one?', 'is it uncertain?' refer to the comment just discussed.\n"
        "- Mention missing true labels ONLY when the user asks about "
        "accuracy or how well the model did; otherwise do not bring it up.\n"
        "- When a question is about a specific comment (e.g. 'why is "
        "comment 3 uncertain?'), answer from that row's prediction.\n"
        "- When the user asks what to do next / what to improve / what to "
        "check, give PRACTICAL BUSINESS recommendations for a shop owner "
        "based on the customer feedback (complaints, repeated problems, "
        "delivery, product quality, price, service, authenticity). Do NOT "
        "recommend model retraining, labeling data, or accuracy "
        "improvement unless the user explicitly asks how to improve the "
        "model itself.\n"
        "- A complaint is a signal to CHECK, not proof: for serious or "
        "unusual complaints, recommend verifying the related order, "
        "product, delivery, or customer-service record first.\n"
        "- If the data does not contain the answer, say it is not "
        "available.\n\n"
        "ACTIVE DATA:\n" + _active_data_text(result)
    )
    messages = [{"role": "system", "content": system}]
    for role, content in _normalize_history(history):
        if len(messages) >= 12:
            break
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    resp = requests.post(
        url + "/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": messages,
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


def _detect_local_llm():
    """Detect a local OpenAI-compatible server (Ollama) on this PC."""
    try:
        r = requests.get("http://127.0.0.1:11434/api/version", timeout=0.5)
        if r.status_code == 200:
            return "http://127.0.0.1:11434/v1", "qwen2.5:3b"
    except Exception:
        pass
    return None, None


def chat(question, result, url=None, key=None, model=None, history=None,
         use_env=True):
    """Answer a user question about `result`.

    Uses an LLM when configured - via explicit `url`/`key`/`model` (UI
    settings) or the AGENT_API_URL / AGENT_API_KEY / AGENT_MODEL env vars,
    or a local Ollama server detected on this PC - and falls back to the
    offline explainer on any failure, so the chat always answers.
    `history` (prior {role, content} messages) is passed to the LLM and
    used by the explainer for follow-up questions. `use_env=False` ignores
    the env vars (the chat page picks providers itself).
    """
    api_url = (url or (os.environ.get("AGENT_API_URL", "") if use_env else "")).rstrip("/")
    api_key = key or (os.environ.get("AGENT_API_KEY", "") if use_env else "")
    api_model = model or (os.environ.get("AGENT_MODEL", "gpt-4o-mini")
                          if use_env else "gpt-4o-mini")
    if api_url and api_key:
        try:
            return _llm_answer(question, result, api_url, api_key, api_model,
                               history=history)
        except Exception:
            pass
    if api_url and not api_key:
        try:
            return _llm_answer(question, result, api_url, "", api_model,
                               history=history)
        except Exception:
            pass
    if not api_url:
        local_url, local_model = _detect_local_llm()
        if local_url:
            try:
                return _llm_answer(question, result, local_url, "",
                                   model or local_model, history=history)
            except Exception:
                pass
    return explain(question, result, history=history)
