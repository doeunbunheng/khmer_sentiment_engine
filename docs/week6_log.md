# Week 6 — Streamlit Dashboard + AI Agent (Log)

Date: 2026-08-07

## Goal

Week 5 hardened the backend (API, security, Docker, OOD guard) but the last
README item was never done: a UI that consumes the `uncertain` flag. Week 6
ships a full Streamlit dashboard on top of the hardened FastAPI server, plus a
**Test data** page that runs the user's own data (pasted or uploaded) through
the live API, and an **Ask the AI agent** chat that explains the results.

## Decisions (agreed before implementation)

1. **Streamlit over static HTML/JS.** Python-only, built-in auth UI, demo-ready.
   (Chose A1 from the Week 6 plan.)
2. **The dashboard talks to the API over HTTP** (never imports `predict.py`
   directly) — so it exercises the exact deployed + secured pipeline.
3. **New API endpoint** `POST /auth/register` — login existed, register didn't.
   Self-registration is hard-coded `role="User"` (can never mint an Admin).
4. **Two test-input modes** (user-requested): paste text (optionally with
   `label|text`) and upload a CSV (text/comment/sentence + label/sentiment/
   polarity). The built-in 989-row benchmark stays as a third option.
5. **AI agent works offline by default** (deterministic explainer built from
   the actual result structs) and upgrades to a real LLM when
   `AGENT_API_URL`/`AGENT_API_KEY` are set — with offline fallback.

## What was built

```
app/
  dashboard.py          Streamlit entry: login/register gate, sidebar nav,
                        conditional Admin page, theme via .streamlit/config.toml
  api_client.py         pure HTTP client (no Streamlit) - token passed per
                        request, never stored on the shared cached client
  unseen_eval.py        parse_input_text / normalize_unseen_df / run_unseen_eval
                        / predict_rows / build_report / build_prediction_summary
  ai_agent.py           explain() offline explainer + chat() with optional LLM
  dashboard_utils.py    cached client + current_auth/require_auth/handle_api_error
  app_pages/
    predict.py          Analyze a comment: badge + confidence, amber `uncertain`
                        panel, aspects + keywords, emotions, consent -> DB save
    unseen_test.py      Test data: paste text / upload CSV / benchmark; progress
                        bar; accuracy/macro-F1/per-class/confusion/by-language/
                        uncertain analysis; JSON report + download
    chat_agent.py       Ask the AI agent: chat over the latest result
    feedback.py         Admin-only feedback table
```

Backend: `src/api.py` gained `POST /auth/register` (rates caps, 409 on
conflict, never Admin role). `requirements.txt` += `streamlit==1.61.1`.

## Tests

- `tests/test_api.py` — 4 register tests (success, never-mints-Admin,
  conflict 409, weak input 422)
- `tests/test_dashboard.py` — 18 tests: ApiClient HTTP layer (bearer +
  consent, URL building, network/HTTP errors, status codes), input parsing,
  output normalization, eval threading + token forwarding, report metrics,
  prediction summary
- `tests/test_ai_agent.py` — 12 tests: intents (positive/negative/uncertain/
  aspects/emotions/metrics/clear), dataset-aware answers, LLM path + fallback
- Full suite: **178/178 pass** (was 144)

## Bugs caught live

1. **Stale module cache in Streamlit** — the running server imported
   `unseen_eval` before `build_prediction_summary` existed; page imports then
   failed. Fix: restart the server (hot reload doesn't clear imported
   packages).
2. **Eval requests 401'd** — `predict_rows`/`run_unseen_eval` called
   `client.predict(text)` without the bearer token. Fixed by threading
   `token=auth["token"]` through. Symptom was "18 rows failed: 401 missing
   bearer token" and empty charts.
3. **Agent gave prediction-style answers for dataset results** — the dataset
   intent (e.g. "what happen on this dataset") fell through to the default
   sentiment explanation ("Confidence is 0%..."). Fixed: dataset contexts
   always route to a benchmark summary (metrics or prediction-overview when
   unlabeled), and "dataset"/"what should I do" are recognized intents.

## Live verification

- API healthy: `{"status":"ok","model":"khmer-sentiment-3class-v2","db":true}`
- Dashboard running at http://localhost:8502 (headless), pages: Analyze /
  Test data / Ask the AI agent / Feedback (Admin).
- Paste-text run of a real corpus (10-31 comments) produced predictions
  (positive/neutral/negative) + per-row table; unlabeled runs show a
  prediction overview (labels enable accuracy/confusion).
- Agent answers for "what happen on this dataset" now explain benchmark
  metrics; "what should I do" gives improvement advice.

## Known limits / next steps

- The offline agent explains with rules; plugging a real LLM key
  (`AGENT_API_URL`/`AGENT_API_KEY`) makes conversations open-ended.
- Full 989-row runs need `API_PREDICT_LIMIT=1000/minute` (documented in the
  page itself).
- Feedback page exposes raw JSONB aspects string — could be expanded into a
  real admin analytics view.