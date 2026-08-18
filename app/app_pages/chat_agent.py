"""Ask the AI agent - discuss the app's prediction results in chat.

The agent talks about the most recent result the user produced: a single
prediction from `Analyze a comment` or a dataset report from `Test data`.
Works offline by default (rule-based explainer that always answers with the
full picture of the result); set AGENT_API_URL + AGENT_API_KEY + AGENT_MODEL
in `.env` (gitignored) to connect a real LLM (any OpenAI-compatible endpoint)
for open-ended chat.
"""

import streamlit as st

from app.ai_agent import chat
from app.dashboard_utils import current_auth

st.title("Ask the AI agent")
st.caption(
    "Ask questions about the latest result the app produced — e.g. \"why is "
    "this positive?\", \"is it uncertain?\", \"what was the comment about?\"."
)

if not (auth := current_auth()):
    st.stop()

context = None
context_label = ""
if (pred := st.session_state.get("last_prediction")):
    context = pred
    context_label = "Last prediction"
elif (ev := st.session_state.get("last_eval")):
    context = ev
    context_label = "Last dataset test"

if context is None:
    st.info(
        "No result to discuss yet. Go to **Analyze a comment** (predict one "
        "comment) or **Test data** (run a benchmark) first, then come back "
        "here and ask about it.",
        icon=":material/forum:",
    )
    st.stop()

if context_label == "Last prediction":
    text = (context.get("text") or "")[:160]
    with st.container(border=True):
        st.markdown(f"**I'm discussing:** the last prediction")
        st.caption(
            f"sentiment `{context.get('sentiment')}` · "
            f"confidence `{context.get('confidence', 0.0):.0%}` · "
            f"uncertain `{bool(context.get('uncertain'))}`"
        )
        if text:
            st.markdown(f"_{text}_")
else:
    with st.container(border=True):
        st.markdown(f"**I'm discussing:** the last dataset evaluation")
        acc = context.get("accuracy")
        ua = context.get("uncertain_analysis") or {}
        unc = ua.get("uncertain_rows")
        if unc is None:
            unc = context.get("uncertain_rows")
        st.caption(
            f"rows `{context.get('rows')}` · "
            + (f"accuracy `{acc:.1%}` · " if acc is not None else "")
            + f"uncertain rows `{unc}`"
        )

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

starter = st.session_state.pop("chat_starter", None)
if starter and not st.session_state.chat_messages:
    st.session_state.chat_messages.append({"role": "user", "content": starter})
    with st.chat_message("user"):
        st.markdown(starter)
    answer = chat(starter, context)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about the result...")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = chat(
        prompt,
        context,
    )
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)