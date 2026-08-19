"""Ask the AI agent - chat about the data the user points it at.

How the agent is linked to the user's data (shown on the page):

  1. The agent is grounded on the ENTIRE latest result: for a dataset run
     that is the summary PLUS every per-row prediction (row number, text,
     sentiment, confidence, `uncertain`, language, aspects, emotions, true
     label when present) - so the user can ask about the dataset overall
     ("why are most comments positive?") or one row ("why is comment 3
     uncertain?") without knowing anything about the model.
  2. The latest dataset test is selected automatically; the user can also
     pick the last single prediction or paste a comment here (run live
     through the API).
  3. Follow-up questions ("why?", "what about this one?") resolve against
     the comment that was just discussed.

Pure logic lives in app/ai_agent.py (unit-tested); this page only wires the
selected context + conversation history into `chat()`.
"""

import os

import streamlit as st

from app.ai_agent import _detect_local_llm, chat
from app.api_client import ApiError
from app.dashboard_utils import current_auth, get_client, handle_api_error

st.title("Ask the AI agent")
st.caption(
    "The agent discusses the prediction result the system just produced - "
    "no model internals needed. Just ask in plain language."
)

if not (auth := current_auth()):
    st.stop()

client = get_client(st.session_state.dashboard_base_url)


def _local_llm_available():
    return _detect_local_llm()[0] is not None


with st.expander("How the agent is linked to your data", expanded=False):
    st.markdown(
        "1. **The latest result is loaded automatically** — after a dataset "
        "test in **Test data**, the agent has every row: sentiment, "
        "confidence, `uncertain`, aspects, emotions, and the true label when "
        "present.\n\n"
        "2. **Ask anything** — overall questions (\"why are most comments "
        "positive?\") or per-row questions (\"why is comment 3 uncertain?\"); "
        "follow-ups like \"why?\" keep the context of the comment you were "
        "just discussing.\n\n"
        "3. **No setup needed** — answers come from your computer's AI "
        "already (no key, no internet). You can optionally switch to "
        "Google Gemini / OpenAI in **AI settings** below, but you don't "
        "have to."
    )

def _configured_providers():
    """Provider entries from .env - each becomes one clickable option."""
    out = []
    if os.environ.get("AGENT_API_URL") and os.environ.get("AGENT_API_KEY"):
        out.append({
            "name": "OpenRouter (.env AI)",
            "url": os.environ["AGENT_API_URL"].rstrip("/"),
            "key": os.environ["AGENT_API_KEY"],
            "model": os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
        })
    for name, url_env, key_env, model_env, default in (
        ("Google Gemini", "GEMINI_API_URL", "GEMINI_API_KEY", "GEMINI_MODEL",
         "gemini-2.5-flash"),
        ("OpenAI GPT", "OPENAI_API_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
         "gpt-4o-mini"),
    ):
        if os.environ.get(url_env) and os.environ.get(key_env):
            out.append({
                "name": name,
                "url": os.environ[url_env].rstrip("/"),
                "key": os.environ[key_env],
                "model": os.environ.get(model_env, "") or default,
            })
    return out


with st.expander("AI settings (optional — the chat already works without this)",
                 expanded=False):
    st.markdown(
        "**You don't need to do anything here.** Your computer's AI is "
        "already answering, with no account, no key, and no internet.\n\n"
        "If your project has AI keys configured (`.env`), the models below "
        "are **ready to use with one click** — no pasting needed.\n\n"
        "អ្នកមិនចាំបាច់ធ្វើអ្វីទេ — អេអាយក្នុងកុំព្យូទ័ររបស់អ្នក"
        "ឆ្លើយបានហើយ មិនត្រូវការគន្លឹះ ឬអ៊ីនធឺណិត។"
    )
    providers = _configured_providers()
    options = ["My computer's AI — no setup needed"]
    options += ["{} — key ready".format(p["name"]) for p in providers]
    if not providers:
        options.append(
            "Google Gemini / OpenAI GPT — keys not configured (see .env)"
        )
    provider = st.radio(
        "Which AI should answer? / តើអេអាយណាគួរឆ្លើយ?",
        options,
        index=0,
        label_visibility="collapsed",
    )
    chosen = None
    if providers:
        for p in providers:
            if provider.startswith(p["name"]):
                chosen = p
                break
    if chosen:
        st.session_state.agent_llm_url = chosen["url"]
        st.session_state.agent_llm_key = chosen["key"]
        st.session_state.agent_llm_model = chosen["model"]
        st.caption(
            f"Active: **{chosen['name']}** (`{chosen['model']}`) — "
            "answers come from this AI, no pasting needed. Falls back to "
            "the offline explainer on any error."
        )
    else:
        for k in ("agent_llm_url", "agent_llm_key", "agent_llm_model"):
            st.session_state.pop(k, None)
        if provider == "My computer's AI — no setup needed":
            st.caption(
                "Active: **your computer's AI** (Qwen 2.5) — no key, no "
                "internet. It answers right now."
            )
        else:
            st.caption(
                "No keys in `.env` yet — add `GEMINI_API_KEY=` or "
                "`OPENAI_API_KEY=` there (developer task, one time) and "
                "this option lights up."
            )

# ---- 1 · choose what the agent should discuss ---------------------------

has_pred = bool(st.session_state.get("last_prediction"))
has_eval = bool(st.session_state.get("last_eval"))

options = []
if has_eval:
    options.append("My last dataset test")
if has_pred:
    options.append("My last prediction")
options.append("A comment I paste here")

default_idx = 0 if has_eval else (1 if has_pred else 0)
st.markdown("### 1 · What data should the agent discuss?")
choice = st.radio(
    "What data should the agent discuss?",
    options,
    label_visibility="collapsed",
    key="chat_choice",
    index=default_idx,
    help=(
        "The agent's answers are based ONLY on the data you select here. "
        "After a dataset test, that result is selected automatically."
    ),
)

if choice == "A comment I paste here":
    with st.form("chat_own_text_form"):
        own_text = st.text_area(
            "Paste a comment",
            height=110,
            max_chars=2000,
            key="chat_own_text",
            placeholder=(
                "Paste any Khmer / English / mixed comment, e.g. "
                "\"ផលិតផលល្អណាស់ តម្លៃសមរម្យ\""
            ),
        )
        analyzed = st.form_submit_button(
            "Analyze and discuss", type="primary", icon=":material/analytics:"
        )
    if analyzed:
        if not own_text.strip():
            st.error("Paste a comment first.")
        else:
            try:
                res = client.predict(own_text, consent=False, token=auth["token"])
            except ApiError as exc:
                handle_api_error(exc, client, auth)
            else:
                st.session_state["chat_own_result"] = res
                st.rerun()
    context = st.session_state.get("chat_own_result")
elif choice == "My last prediction":
    context = st.session_state.get("last_prediction")
else:
    context = st.session_state.get("last_eval")

if context is None:
    st.info(
        "No data to discuss yet. Either paste a comment above, or go to "
        "**Analyze a comment** (predict one comment) / **Test data** (run a "
        "benchmark) first — then come back here and ask about it.",
        icon=":material/forum:",
    )
    st.stop()

# ---- 2 · show exactly what the agent is grounded on ---------------------

is_dataset = context.get("type") == "dataset"
st.markdown("### 2 · The agent is grounded on this data")
with st.container(border=True):
    if is_dataset:
        st.markdown("**Last dataset test**")
        acc = context.get("accuracy")
        ua = context.get("uncertain_analysis") or {}
        unc = ua.get("uncertain_rows")
        if unc is None:
            unc = context.get("uncertain_rows")
        dist = context.get("distribution") or {}
        dist_bits = ", ".join(
            f"{v} {k}" for k, v in dist.items() if v
        )
        st.caption(
            f"rows `{context.get('rows')}` · "
            + (f"accuracy `{acc:.1%}` · " if acc is not None else "")
            + f"uncertain rows `{unc}`"
            + (f" · {dist_bits}" if dist_bits else "")
        )
        preds = context.get("predictions")
        if isinstance(preds, list) and preds:
            st.caption(
                f"the agent can also discuss all {len(preds)} rows "
                "individually (e.g. \"why is comment 3 uncertain?\")"
            )
        if context.get("dataset"):
            st.caption(f"source `{context['dataset']}`")
    else:
        text = (context.get("text") or "").strip()
        st.markdown(
            f"**Comment**: {text[:200]}" + ("…" if len(text) > 200 else "") or "—"
        )
        st.caption(
            f"sentiment `{context.get('sentiment')}` · "
            f"confidence `{context.get('confidence', 0.0):.0%}` · "
            f"uncertain `{bool(context.get('uncertain'))}` · "
            f"language `{context.get('language')}`"
        )

# ---- 3 · ask about it ----------------------------------------------------

st.markdown("### 3 · Ask about this data")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

ctx_key = (
    ("dataset", str(context.get("rows")), context.get("dataset"))
    if is_dataset
    else ("pred", (context.get("text") or "")[:80])
)
if ctx_key != st.session_state.get("_chat_ctx_key"):
    st.session_state["_chat_ctx_key"] = ctx_key
    st.session_state.chat_messages = []

if is_dataset:
    starters = [
        "What happened in this dataset?",
        "Why are most comments positive?",
        "Show me the uncertain comments",
        "What are customers complaining about?",
        "Is the model doing well?",
        "What should I do next?",
    ]
else:
    starters = [
        "Why is this the result?",
        "Is it uncertain?",
        "What was the comment about?",
        "How confident is the model?",
        "Explain simply",
    ]
picked = st.pills(
    "Quick questions", starters, key="chat_pill", label_visibility="collapsed"
)
if picked and picked != st.session_state.get("_last_chat_pill"):
    st.session_state["_last_chat_pill"] = picked
    st.session_state.chat_messages.append({"role": "user", "content": picked})

starter = st.session_state.pop("chat_starter", None)
if starter:
    st.session_state.chat_messages.append({"role": "user", "content": starter})

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about this data — e.g. \"why is it positive?\"...")
question = None
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    question = prompt
elif picked and picked == st.session_state.get("_last_chat_pill"):
    question = picked
elif starter:
    question = starter

if question:
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
    history = st.session_state.chat_messages[:-1]
    answer = chat(
        question, context,
        url=st.session_state.get("agent_llm_url") or None,
        key=st.session_state.get("agent_llm_key") or None,
        model=st.session_state.get("agent_llm_model") or None,
        history=history,
        use_env=False,
    )
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
