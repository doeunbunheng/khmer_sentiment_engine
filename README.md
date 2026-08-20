# Khmer Sentiment Engine

3-class sentiment (Positive / Negative / Neutral) + aspect analysis for
Khmer, English, and code-switched consumer comments.

**Status: Week 6 complete (+ AI agent upgrade)** — Streamlit dashboard (`app/`) on top of the
hardened API: login/register, live prediction with the **uncertain** OOD state,
a **Test data** page (paste text / upload CSV / built-in 989-row benchmark run
through the live API), an **Ask the AI agent** chat that explains each result
(domain-aware offline explainer + local PC AI + optional one-click cloud
providers), Admin feedback view, and a new
`POST /auth/register` endpoint. 233/233 tests pass.


## Executive summary — what was done (read this first)

**The project:** "Khmer Sentiment Engine" — a machine-learning system that
automatically reads Khmer, English, and mixed (code-switched) customer
comments (e.g. from Facebook, Shopee, Google Reviews) and tells a business:
**(1)** whether each comment is positive / negative / neutral, **(2)** what it
is about (5 business aspects: Price, Service, Product Quality, Authenticity,
Delivery + 8 emotions), **(3)** an honest "not sure" flag when the model's
confidence is low, and **(4)** an AI chat assistant the owner can ask questions
about the results in plain Khmer or English.

**Built over 6 weeks** (July–August 2026), everything is in this repository and
on GitHub (https://github.com/doeunbunheng/khmer_sentiment_engine):

| Week | What was done | Result |
|---|---|---|
| 1 | Collected and cleaned ~18,771 labeled Khmer comments; split into train / validation / test | Clean dataset + 3 stratified splits |
| 2 | Built sentiment detection (Khmer/English/mixed) with automatic fallback when the cloud API failed | 
| 3 | Added PostgreSQL database + secure login/register + anonymized feedback storage (with user consent) | 
| 4 | Fine-tuned our own 3-class model (xlm-roberta-base, on this PC's GPU) — the old pipeline scored 0.6821, the new model **0.8211** accuracy; checked that a more complex "stacking ensemble" gave no gain (documented honestly) | 
| 5 | Added aspect analysis (5 business aspects + 8 emotions), tested the model on a **truly unseen** news/politics corpus (found a real weakness: 0.4165 accuracy → retrained a mixed-domain "v2" model → **0.8241** on unseen data), built a FastAPI web server, secured it (login tokens, rate limits, admin-only feedback), and packaged it in Docker | 
| 6 | Built a complete **web dashboard** (Streamlit): login, live analysis, testing user's own data, an **AI chat agent** that explains results (offline explainer / local PC AI / one-click Gemini-GPT), and admin feedback view |
| 6b | Upgraded the AI agent to work on **any dataset** (not just shopping): it discovers topics from the text itself, only gives shopping advice for shopping data, and switches between 3 AI providers with one click | 

**Key numbers a supervisor cares about:**

- Final model: `xlm-roberta-base` fine-tuned, 3-class, "v2" in production
- Test accuracy (in-domain, 1,878 rows): **0.8333**
- Accuracy on unseen data (989 held-out news/politics rows): **0.8241**
  (fixed a real "domain shift" problem: old model 0.4014)
- Live web API gives **exactly the same results** as offline (989/989 rows identical)
- **233/233 automated tests pass** — quality gate for every change
- Security: tokens, rate limits, login lockout, admin-only access, no personal
  data in logs, consent-gated feedback storage















---

## Executive summary — what was done (read this first)

**The project:** "Khmer Sentiment Engine" — a machine-learning system that
automatically reads Khmer, English, and mixed (code-switched) customer
comments (e.g. from Facebook, Shopee, Google Reviews) and tells a business:
**(1)** whether each comment is positive / negative / neutral, **(2)** what it
is about (5 business aspects: Price, Service, Product Quality, Authenticity,
Delivery + 8 emotions), **(3)** an honest "not sure" flag when the model's
confidence is low, and **(4)** an AI chat assistant the owner can ask questions
about the results in plain Khmer or English.

**Built over 6 weeks** (July–August 2026), everything is in this repository and
on GitHub (https://github.com/doeunbunheng/khmer_sentiment_engine):

| Week | What was done | Result |
|---|---|---|
| 1 | Collected and cleaned ~18,771 labeled Khmer comments; split into train / validation / test | Clean dataset + 3 stratified splits |
| 2 | Built sentiment detection (Khmer/English/mixed) with automatic fallback when the cloud API failed | 35/35 tests pass |
| 3 | Added PostgreSQL database + secure login/register + anonymized feedback storage (with user consent) | 52/52 tests pass |
| 4 | Fine-tuned our own 3-class model (xlm-roberta-base, on this PC's GPU) — the old pipeline scored 0.6821, the new model **0.8211** accuracy; checked that a more complex "stacking ensemble" gave no gain (documented honestly) | 53/53 tests pass |
| 5 | Added aspect analysis (5 business aspects + 8 emotions), tested the model on a **truly unseen** news/politics corpus (found a real weakness: 0.4165 accuracy → retrained a mixed-domain "v2" model → **0.8241** on unseen data), built a FastAPI web server, secured it (login tokens, rate limits, admin-only feedback), and packaged it in Docker | 144/144 tests pass |
| 6 | Built a complete **web dashboard** (Streamlit): login, live analysis, testing user's own data, an **AI chat agent** that explains results (offline explainer / local PC AI / one-click Gemini-GPT), and admin feedback view | 178/178 tests pass |
| 6b | Upgraded the AI agent to work on **any dataset** (not just shopping): it discovers topics from the text itself, only gives shopping advice for shopping data, and switches between 3 AI providers with one click | **233/233 tests pass** |

**Key numbers a supervisor cares about:**

- Final model: `xlm-roberta-base` fine-tuned, 3-class, "v2" in production
- Test accuracy (in-domain, 1,878 rows): **0.8333**
- Accuracy on unseen data (989 held-out news/politics rows): **0.8241**
  (fixed a real "domain shift" problem: old model 0.4014)
- Live web API gives **exactly the same results** as offline (989/989 rows identical)
- **233/233 automated tests pass** — quality gate for every change
- Security: tokens, rate limits, login lockout, admin-only access, no personal
  data in logs, consent-gated feedback storage

**What a supervisor can do to see it working:** run the API + dashboard
(commands in "Run it" below), log in with the demo admin account, paste any
Khmer/English comment, and click "Ask the AI agent" to chat about the results.

---

## Project reference — for the supervisor

**Khmer Sentiment Engine** — a complete machine-learning system that
automatically reads Khmer, English, and mixed (code-switched) customer
comments and tells a business what customers feel and why.

### Problem it solves

Businesses collect thousands of comments on Facebook / Shopee / Google
Reviews, but nobody has time to read them all. This engine replaces
manual reading with: **(1)** a sentiment score for every comment
(positive / negative / neutral), **(2)** what the comment is about (5
business aspects: Price, Service, Product Quality, Authenticity,
Delivery, + 8 emotions), **(3)** a "not sure" flag when the model is
unsure (confidence < 90%), and **(4)** an AI assistant the owner can ask
questions about the results in plain language — in Khmer or English.

### Key results (honest numbers)

| Item | Result |
|---|---|
| Final model | xlm-roberta-base fine-tuned, 3-class, **v2 in production** |
| In-domain test accuracy (1,878 rows) | **0.8333** |
| Unseen-domain accuracy (989 held-out news/politics rows) | **0.8241** (old v1: 0.4014 — domain shift fixed) |
| Baseline before the fine-tuned model | 0.6821 |
| Calibration | ECE 0.086 (val); uncertainty threshold 0.90 confirmed |
| Live API = offline parity | 989/989 rows identical (0 errors) |
| Tests | **233/233 passing** |

### Architecture (3 layers)

```
Streamlit dashboard  →  FastAPI server  →  PostgreSQL
(app/)                 (src/api.py)       (feedback loop, users)
                            │
                    Sentiment model (xlm-r 3-class v2)
                    Aspects (rules) + Emotions (songhieng)
                    AI agent (offline explainer / local Ollama /
                              cloud Gemini/GPT — one click)
```

### What the user can do today (demo)

1. **Login / Register** — secure accounts (`demo_admin` / `132336BV132336`)
2. **Analyze a comment** — paste any Khmer/English comment → sentiment,
   confidence, aspects, emotions; "not sure" state when ambiguous
3. **Test data** — paste text, upload a CSV, or run the built-in 989-row
   benchmark through the live API → accuracy, confusion matrix,
   uncertain analysis, downloadable report
4. **Ask the AI agent** — chat about any result: "why are comments
   negative?", "show me the uncertain ones", "what should I do next?" —
   answers adapt to the actual dataset (works for shops AND for
   education/news/other data), powered by the PC's local AI by default
   (no key, no internet) with optional one-click Gemini/GPT
5. **Feedback** (Admin) — see every saved prediction

### Security & engineering quality

- HMAC-signed bearer tokens, rate limits, login lockout, Admin-only
  feedback, CORS lockdown, no-PII logs
- Anonymized feedback storage with consent gate
- Calibration-driven design: low-confidence comments are flagged, never
  silently guessed
- Verified generalization on a truly unseen corpus (0 overlap with
  training data) — documented limitation: the model is strongest on
  consumer-review-style text

### Resources & references (models, data, tools)

Everything below is used directly in this project — base models, training
data, and libraries:

| Resource | What it is | Where we use it |
|---|---|---|
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | Multilingual transformer (100 languages, incl. Khmer) — our **base model** | Fine-tuned → `models/khmer-sentiment-3class-v2` (the production sentiment model) |
| [5oni7a/Khmer-Profanity](https://huggingface.co/datasets/5oni7a/Khmer-Profanity) | ~18.7k labeled Khmer food-review comments — **main training data** | Cleaned → `data/labeled/external_processed.csv` (18,771 rows) |
| [ye-kyaw-thu/kh-polarity](https://github.com/ye-kyaw-thu/kh-polarity) | iSAI-NLP annotated Khmer polarity corpus (news/politics) — **unseen benchmark** | Mixed 7,905 rows into training (v2); 989 rows held out for external validation |
| [songhieng/khmer-xlmr-base-sentimental-multi-label](https://huggingface.co/songhieng/khmer-xlmr-base-sentimental-multi-label) | Khmer multi-label emotion model (8 emotions) | Cached at `models/khmer-aspects-multilabel/` → aspect analysis |
| [tykea/khmer-text-sentiment-analysis-roberta](https://huggingface.co/tykea/khmer-text-sentiment-analysis-roberta) | Khmer 2-class sentiment model (positive/negative) | Fallback path (`predict_fallback`) when the production model cannot load |
| [khmer-nltk](https://github.com/VietAI/khmer-nltk) | Khmer word segmentation + POS | `src/preprocessing/segment.py`; the AI agent's topic discovery |
| [Hugging Face transformers](https://github.com/huggingface/transformers) | Model training / inference framework | All model loading, fine-tuning (`train_3class.py`), evaluation |
| [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) | Python web API | `src/api.py` — the production prediction server |
| [Streamlit](https://streamlit.io/) | Python dashboard framework | `app/` — the user-facing dashboard |
| [PostgreSQL](https://www.postgresql.org/) | Database | Users, anonymized feedback, analysis records (`src/db/schema.sql`) |
| [Ollama](https://ollama.com) + [Qwen 2.5 3B](https://ollama.com/library/qwen2.5) | Local LLM for the chat assistant | Auto-detected local AI (no key/internet) in "Ask the AI agent" |

**Methodology references** — the design decisions in this project follow
established practice:

- Confidence-based abstention (`uncertain` when conf < 0.90) — the
  model never silently guesses; see the **Calibration & robustness**
  section (ECE 0.086, ≥0.90 bin = 0.91 acc).
- External / out-of-domain validation — the v1 → v2 retrain story
  (0.4014 → 0.8241 on unseen data) is documented in the **Week 5**
  section and `docs/week5_log.md`.
- Project repository: https://github.com/doeunbunheng/khmer_sentiment_engine

### Run it

```bash
./.venv/Scripts/python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000  # API
./.venv/Scripts/python.exe -m streamlit run app/dashboard.py                      # dashboard
./.venv/Scripts/python.exe -m pytest tests -q                                     # 233 tests
```

---

## Quick start

```bash
# 1. Environment
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Data splits (reads data/labeled/*.csv, writes data/splits/)
.venv\Scripts\python scripts\prepare_splits.py

# 3. Pre-flight check + tests (always before a long run)
.venv\Scripts\python scripts\preflight_check.py
.venv\Scripts\python -m pytest tests -v

# 4. Database (local PostgreSQL, or: docker compose up -d db)
psql -U postgres -h localhost -f src/db/schema.sql

# 5. Predict from a file / stdin / paste
.venv\Scripts\python scripts\predict_demo.py notes.txt
```

**Rule of thumb for every experiment:** `preflight_check.py` → smoke run
(`--smoke` / `--max-rows 200`) → full run. Smoke takes ~2 min and catches bugs
before you waste a 10–50 min GPU run.

---

## Completed status — overall

| Week | Deliverable | Result |
|---|---|---|
| 1 | Setup + data prep | 18,771 labeled rows, stratified splits (15,016 / 1,877 / 1,878) |
| 2 | Detection + sentiment + resilience | tykea live predictions, neutral rule, API→local fallback, **35/35 tests** |
| 3 | Postgres + secure feedback loop | Login/Register, anonymization, consent gate, **52/52 tests** |
| 4 | 3-class fine-tune + integration | Baseline 0.6821 → fine-tuned **0.8211** test acc, **53/53 tests** |
| Hardening | Calibration + workflow | ECE 0.083, uncertain rule < 0.90, smoke modes, preflight, 17 script tests → **70/70 tests** |
| 5 | Aspects + external validation | 5 business aspects (rules) + 8 emotions (songhieng), wired into pipeline, **101/101 tests**; unseen-data check → 0.4165 acc |
| 5b | v2 mixed-domain retrain | +7,905 kh-polarity train rows → in-domain **0.8333**, unseen **0.8241** acc; v2 in production |
| 5c | Local API deployment | FastAPI (`src/api.py`) serving the pipeline; 989 unseen rows via API = **0.8241** (exact parity); **105/105 tests** |
| 5d | API security hardening | HMAC tokens + rate limits + lockout + Admin-only feedback + CORS + no-PII logs; 989-row parity **0.8241** kept; **115/115 tests** |
| 5e | Docker + polish + OOD guard | `docker compose up` (api+db) verified end-to-end; aspect dictionaries grown from real corpus (+30 keywords); `uncertain` flag (conf < 0.90) in every prediction; calibration re-checked on v2 (ECE 0.086/0.113); **144/144 tests** |

---

## Week 1 — Setup & Data Prep

| Gate | Result |
|---|---|
| Environment | Python 3.14 venv, `requirements.txt`, `.env` / `config.yaml` |
| Cleaning | `clean.py`: URLs, emails, mentions, hashtags, Unicode NFC, whitespace |
| Segmentation | `segment.py`: khmer-nltk word segmentation (regex fallback) |
| Labeled data | `data/labeled/external_processed.csv` — 18,771 unique labeled rows |
| Split 80/10/10 (stratified) | train 15,016 / val 1,877 / test 1,878 (CSV + `splits_report.json`) |

## Week 2 — Detection, Sentiment, Resilience

| Gate | Result |
|---|---|
| Language detection | `khmer` / `english` / `mixed` / `unknown` (Unicode regex) |
| Sentiment (tykea model) | Live: Khmer, English (auto EN→KM translate), mixed — all return (label, score) |
| Neutral rule | score < 0.60 → `neutral` (3rd class without retraining) |
| API resilience | HF API DNS-dead globally → automatic local fallback (same model, cached) |
| `pytest tests/ -v` | 35/35 passed |

## Week 3 — Postgres + Secure Feedback Loop

| Gate | Result |
|---|---|
| Login / Register | Secure: bcrypt hash (pgcrypto), SQL `register_user()` / `login_user()`, roles `Admin` / `User` |
| Anonymization | Emails, phones, Khmer IDs, URLs, mentions, names → `[ANONYMIZED]` before save; trigger-enforced |
| Consent gate | `consent_granted` required — no save without consent (Python + DB trigger) |
| Feedback loop | `predict_and_save()` → anonymized row persisted for every agreed prediction |
| Postgres | `khmer_sentiment` DB: `users`, `user_profiles`, `user_feedback`, `analysis_records` |
| `pytest tests/ -v` | 52/52 passed |

## Week 4 — 3-Class Fine-Tune + Integration

Full step-by-step record: [`docs/week4_log.md`](docs/week4_log.md). Summary:

### Phase 0 — Environment (27 min)
Fresh Python 3.14 venv + CUDA torch `2.13.0+cu130` on RTX 4060; old venv backed up to `.venv-backup`.

### Phase 1 — Baseline to beat (5m 22s)
`scripts/evaluate_baseline.py` — old pipeline (tykea + 0.60 neutral rule) on `test.csv`:

| Metric | Value |
|---|---|
| Accuracy | **0.6821** |
| Macro-F1 | **0.6071** (neg 0.73 / neu 0.33 / pos 0.76) |
| By language | khmer 0.694 · english 0.458 · mixed n=7 |

Key finding: the threshold rule can't detect neutral (F1 0.33) and English is nearly random → a real 3-class model is required. → `reports/baseline.json`

### Phase 2 — Train the 3-class model (11 min)
`scripts/train_3class.py` — **xlm-roberta-base**, neutral oversampled 2.5×, class-weighted loss, early stop on val macro-F1 (3 epochs, fp16, seed 42). Saved → `models/khmer-sentiment-3class/`. Val results → `reports/phase2_val.json`:

| Metric | Value |
|---|---|
| Accuracy | **0.8503** |
| Macro-F1 | **0.8430** |
| Per-class F1 | negative 0.841 · neutral **0.815** · positive 0.873 |
| By language | km 0.855 · en 0.766 · code-switched 0.833 |

Headline: neutral F1 jumped 0.33 → 0.815, English acc 0.458 → 0.766 — test set untouched.

### Phase 3 — Stacking ensemble (51.6 min) — **no gain, documented**
`scripts/stack_phase3.py` — 5-fold OOF probabilities (xlm-r 3-class + tykea pseudo-probs + language one-hot) → LogisticRegression meta-learner:

| Model | Val acc | Macro-F1 |
|---|---|---|
| Phase 2 single model | **0.8503** | **0.8430** |
| xlm-r 5-fold OOF (no meta) | 0.8476 | 0.8413 |
| Stack (xlm-r + tykea + lang) | 0.8450 | 0.8370 |

Conclusion: **stacking adds no value** — tykea features actively hurt. Phase 4 uses the Phase 2 single model. OOF features cached → `reports/phase3_oof.npz`.

### Phase 4 — One-shot final test
`scripts/evaluate_final.py` — the **untouched** `test.csv` (1,878 rows) → `reports/phase4_test.json`:

| Metric | Value |
|---|---|
| Accuracy | **0.8211** |
| Macro-F1 | **0.8156** (neg 0.809 / neu 0.796 / pos 0.842) |
| By language | km 0.8259 · en 0.7188 · code-switched 1.0 (n=7) |

vs baseline: **+0.139 accuracy, +0.209 macro-F1**.

### Phase 5 — Integration
- `config.yaml` → `local_model_path: models/khmer-sentiment-3class`, `local_max_length` 256
- `src/models/local_model.py` → production `predict` (3-class) + `predict_fallback` (tykea)
- `src/predict.py` → **translation + 0.60 threshold rule removed**; model handles Khmer/English/mixed natively
- Live check: Khmer pos 0.986 · Khmer neg 0.994 · English pos 0.990 · mixed pos 0.817
- **53/53 tests passed**

### English experiment — direct vs translate (DONE)

`scripts/compare_en_direct_vs_translate.py` — 96 English `test.csv` rows, same model:

| Metric | English DIRECT | EN→KM translated |
|---|---|---|
| Accuracy | **0.7188** | 0.6875 |
| Macro-F1 | **0.6594** | 0.6359 |
| Speed | **0.9 ms/row** | 290 ms/row |

- **Direct is ~329× faster** (0.085 s vs 27.8 s for 96 rows)
- Translation flipped 13/96 rows (13.5%) to wrong labels
- Overall effect on the test set: 0.8211 (direct) vs 0.8195 (translate)
- **Decision: keep the no-translation pipeline** → `reports/english_direct_vs_translate.json`

### Full results — training / validation / testing

| Set | Rows | Accuracy | Macro-F1 |
|---|---|---|---|
| Training (5-fold OOF, honest\*) | 15,016 | 0.8358 | 0.8272 |
| Validation (early-stopped) | 1,877 | 0.8503 | 0.8430 |
| Testing (untouched, final) | 1,878 | 0.8211 | 0.8156 |

Per-class F1:

| Set | negative | neutral | positive |
|---|---|---|---|
| Train | 0.8264 | 0.7950 | 0.8602 |
| Val | 0.8412 | 0.8148 | 0.8731 |
| Test | 0.8094 | 0.7958 | 0.8418 |

By language (accuracy):

| Set | khmer | english | code-switched |
|---|---|---|---|
| Train | — | — | — |
| Val | 0.8548 (n=1777) | 0.7660 (n=94) | 0.8333 (n=6) |
| Test | 0.8259 (n=1775) | 0.7188 (n=96) | 1.0000 (n=7) |

\* OOF = each train row predicted by a fold model that never trained on it (no leakage) — the honest train number. In-sample accuracy (~0.97+) is meaningless.

Reading the numbers:

1. **No overfitting** — train ≈ val ≈ test (0.84 / 0.85 / 0.82), healthy ~0.03 gap
2. **Test slightly below val**, as expected — early stopping tuned on val
3. **Neutral is the hardest class** (F1 ~0.80) — inherently ambiguous
4. **English is the weak spot** (test 0.719) — only 96 rows; more EN data would help

Sources: `reports/phase2_val.json`, `reports/phase4_test.json`, `reports/phase3_oof.npz`

---

## Week 6 — Streamlit dashboard + AI agent (done)

`app/` — a full UI on top of the hardened API, run with:

```bash
./.venv/Scripts/python.exe -m streamlit run app/dashboard.py        # http://localhost:8501
```

| Page | What it does |
|---|---|
| Log in / Create account | New `POST /auth/register` endpoint (self-registration can never mint an Admin); tokens stored per session |
| Analyze a comment | Type/example Khmer/English/mixed text → sentiment badge + confidence, **amber "not sure" panel when `uncertain`** (conf < 0.90 — the Week 5 leftover), business aspect hits + matched keywords, active emotions, consent checkbox → saves to DB |
| Test data | Benchmark the **live API** on user data: **(1) paste text** (optionally with `negative\|text` labels for accuracy) or **(2) upload a CSV** (text/comment/sentence + label/sentiment/polarity columns) or the built-in 989-row held-out set. Threaded with a progress bar → accuracy, macro-F1, per-class F1, confusion matrix, by-language, uncertain analysis; report JSON saved/downloadable |
| Ask the AI agent | Chat that discusses the latest result ("why positive?", "is it uncertain?", "what was it about?") — **domain-aware offline explainer** (works on any dataset, no key needed), **local PC AI** (Ollama + qwen2.5:3b, auto-detected), or **one-click cloud providers** configured in `.env` (`AGENT_*` / `GEMINI_*` / `OPENAI_*`) — with offline fallback on any error |
| Feedback | Admin-only table of stored feedback rows |

Components: `app/api_client.py` (pure HTTP client, token passed per request),
`app/unseen_eval.py` (evaluation logic, no Streamlit), `app/ai_agent.py`
(explainer + optional LLM), `app/dashboard_utils.py` (cached client + auth).

Notes: the full 989-row run needs the API started with
`API_PREDICT_LIMIT=1000/minute` (default 120/min); one bug fixed during the
week — evaluation requests now send the bearer token (were 401-ing).

---

## Week 6b — AI agent: any-dataset awareness + one-click AI providers (done)

The chat assistant was upgraded so it behaves like a general-purpose AI
(Gemini/GPT style) **grounded on the user's own prediction result** — even
when the dataset has nothing to do with the 5 built-in business aspects.

### Dataset-adaptive answers (offline explainer)

- **Topic discovery from the text itself** — when the fixed business
  aspects don't hit, the agent segments the comments (khmer-nltk, regex
  fallback for English), removes stopwords, and names the most repeated
  words as the real topics. Education / politics / news / restaurant
  datasets get answers about *their own words* (e.g. `ព័ត៌មាន`, `រៀន`).
- **One-sentence class summaries** — "why are comments negative?" answers
  with one line ("In one sentence: these 13 negative comments are mostly
  about …") + 3 examples + a **What to do** action line, instead of dumping
  every comment. Long pasted inputs are truncated, never echoed back.
- **Commerce-aware advice gating** (`_is_commerce_data`) — shopping advice
  ("check the supplier", "packing and shipping", "wrong orders") is only
  given when the comments clearly talk about buying/selling/delivering
  (ដឹកជញ្ជូន, អីវ៉ាន់, លក់, ទិញ, delivery, product…). Ambiguous words
  (ពិត "real", តម្លៃ "value", គុណភាព "quality") no longer trigger it —
  an education dataset gets neutral investigation advice instead of
  product-returns advice.
- **Intent routing** — business advice vs technical ML advice are
  separated: "what should I do next?" → shop-owner actions; "how to
  improve the model?" → retrain/labeling/accuracy guidance. Advice words
  (suggest / idea / recommend / advice) append the action block to class
  answers. Follow-ups ("why?", "what about this one?") resolve to the
  comment just discussed via conversation history.

### Bug fixed — Khmer topic discovery silently failed

`isalnum()` returns `False` for Khmer words containing combining marks
(៌ ្ ំ …), so every Khmer content word was filtered out and the agent
claimed "no topic repeats" on Khmer-only data. Fixed by keeping tokens
with at least one letter; regression tests assert Khmer words with marks
survive segmentation.

### Three AI tiers — one-click switching in the UI

| Tier | Setup | When used |
|---|---|---|
| ① Offline explainer | none | always available; used on any LLM error |
| ② Local PC AI | install [Ollama](https://ollama.com) + `ollama pull qwen2.5:3b` | **auto-detected** at `http://127.0.0.1:11434/v1` — no key, no internet |
| ③ Cloud providers | keys in `.env` | clickable options: "OpenRouter (.env AI)", "Google Gemini — key ready", "OpenAI GPT — key ready" |

`.env` provider slots (keys never appear in the UI or the repo):

```ini
AGENT_API_URL=https://openrouter.ai/api/v1
AGENT_API_KEY=sk-or-v1-...        # any OpenAI-compatible key
AGENT_MODEL=nvidia/nemotron-3.5-lightning:free
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

The UI's **AI settings** panel explains in plain language (English +
ខ្មែរ) that nothing needs to be done — the local AI already answers. Users
pick a provider by clicking a radio button; URL/model come from `.env`.

**233/233 tests pass** — including: per-row questions, follow-ups,
business-vs-technical advice, show-all lists, text-term fallback, Khmer
combining-mark segmentation, commerce gating (education dataset gets no
shopping advice), and local-LLM detection (monkeypatched).

---

## Week 5 — Aspects + External Validation

Aspect analysis adds "what is the comment about" on top of sentiment:

### Business aspects — keyword rules (no model)

`src/models/aspects.py::detect_business_aspects()` — multi-label detection of
the 5 business aspects via Khmer + English keyword dictionaries (no download,
no GPU, explainable — matched keywords are returned):

| Aspect | Example matched words |
|---|---|
| Price | ថ្លៃ · តម្លៃ · ថោក · price · cost · discount |
| Service | សេវា · បុគ្គលិក · staff · support |
| Product Quality | គុណភាព · ផលិតផល · quality · durable |
| Authenticity | ដើម · ក្លែងក្លាយ · authentic · counterfeit |
| Delivery | ដឹកជញ្ជូន · ចែកចាយ · delivery · parcel |

Output: `{aspect: {"hit": bool, "keywords": [matched]}}` — a comment can hit
multiple aspects at once.

### Emotions — songhieng multi-label model

`src/models/aspects.py::predict_emotions()` — `songhieng/khmer-xlmr-base-sentimental-multi-label`
(8 emotions: Anger, Anticipation, Disgust, Fear, Joy, Optimism, Sadness,
Surprise). Sigmoid over 8 logits (multi-label), active = prob ≥ 0.5.

- **Lazy-loaded** (`lru_cache`) — first call downloads ~1.2 GB, then caches to
  `models/khmer-aspects-multilabel/` (config `aspect_model.local_path`)
- Label-count guard: config labels must equal `model.config.num_labels`
- Every aspect failure degrades to empty dicts — sentiment never breaks

### Integration

- `predict_sentiment()` now returns `aspects: {business_aspects, emotions}`
  (empty for blank text; failure-safe)
- `predict_and_save()` stores the whole thing in the existing `aspects` JSONB column
- `predict_demo.py` prints it with no changes
- **31 new tests** (rules, threshold, fake-model emotions, failure paths, DB
  round-trip) — the songhieng model is mocked so the suite never downloads
- Live check: Khmer + English comments → correct aspects + emotions; model
  cached locally, second run ~17 s no download

### External validation on unseen data — kh-polarity (honest result)

The in-domain numbers (0.82) come from data that all originates from one
Hugging Face food-review dataset (`5oni7a/Khmer-Profanity`). To check
real-world generalization, we found and tested a **truly unseen** corpus:

- **Source:** `ye-kyaw-thu/kh-polarity` (GitHub, iSAI-NLP paper) — human-
  annotated Khmer polarity corpus, `sentence ||| keyword ||| polarity`
- Parsed to `data/external_kh_polarity.csv` — **9,882 clean sentences**
  (positive 5,762 / negative 3,203 / neutral 917), domain: news/politics
- **Overlap check: 0 rows** in common with train/val/test (18,771) — genuine
  out-of-domain test
- Eval script: `scripts/evaluate_external.py` (`--max-rows N` smoke,
  `--aspects` for aspect hit stats) → `reports/external_kh_polarity.json`

| Metric | Value |
|---|---|
| Accuracy | **0.4165** |
| Macro-F1 | **0.4216** (neg 0.547 / neu 0.236 / pos 0.483) |
| By language | khmer 0.4369 (n=8697) · mixed 0.2667 (n=1185) |

**Reading the result:**

1. **Domain shift is real** — the model massively over-predicts `neutral`
   (5,717 predicted vs 917 true); news/political text is long, formal and
   annotated differently from short food reviews
2. **The 0.8211 number is valid only for food-review-style text** — exactly
   why an external test matters
3. Aspect rules hold up on unseen text: Authenticity 20.8% hit rate,
   Price 7.1%, Delivery 5.4%

**Options going forward** (partially done): (b) ✅ retrain with part of
kh-polarity mixed into the training data — done, see below; (a) keep this as
a hard benchmark for every future fine-tune; (c) document the limitation in
the UI (show low-confidence "uncertain" state more aggressively on
out-of-domain text).

### v2 — mixed-domain retrain (done, in production)

**Diagnosis (why the old model failed on unseen data):** the old model was
**confidently wrong** out of domain — 72% of kh-polarity rows got confidence
≥ 0.90 with only 0.37 accuracy (vs 0.915 in-domain). Not truncation (mean
49 tokens) and not thresholds — it had overfit to the food-review register
and collapsed long formal text toward `neutral`.

**Fix:** retrained with kh-polarity training rows mixed in
(`train_3class.py --extra-train/--extra-val`, 22,921 train / 2,865 val rows,
same hyperparameters, seed 42). kh-polarity split 80/10/10 stratified →
`data/external_splits/` (train_mix 7,905 / val_mix 988 / **test_ext 989 —
held out, never trained on**).

**Results** (`scripts/evaluate_final.py --model-dir`, `evaluate_external.py
--dataset test_ext.csv`):

| Benchmark | Old model (v1) | **v2 (production)** |
|---|---|---|
| In-domain test.csv (1,878) | 0.8211 / 0.8156 | **0.8333 / 0.8291** |
| Unseen kh-polarity held-out (989) | 0.4014 / 0.4109 | **0.8241 / 0.7543** |
| kh-polarity full corpus (9,882)¹ | 0.4165 / 0.4216 | 0.9171 / 0.8982 |

¹ informational only — includes rows v2 trained on.

Per-class F1 on unseen held-out (v2): negative **0.829** · neutral **0.570** ·
positive **0.864** — the neutral collapse is gone (old: 0.14 precision,
9,017 over-predicted neutrals).

**Conclusion:** mixed-domain training fixed the generalization gap with **no
in-domain regression** — both benchmarks improved. v2 is the production model
(`config.yaml` → `local_model_path: models/khmer-sentiment-3class-v2`);
v1 kept at `models/khmer-sentiment-3class` for comparison.
Reports: `reports/phase2_val_v2.json`, `phase4_test_v2.json`,
`external_test_ext_old.json` / `external_test_ext_v2.json`.

### Local API deployment (done)

`src/api.py` — FastAPI server serving the exact production pipeline
(`predict_and_save()`), models lazy-load on first request:

- `GET /health` → model + DB status (open)
- `POST /auth/login` → `{username, password}` → signed token (rate-limited)
- `POST /predict` → bearer token + `{text, user_id?, consent?, names?}` → full
  prediction (sentiment, confidence, language, aspects, `saved_id?`)
- `GET /feedback` → bearer token + **Admin** role → latest feedback rows
- `GET /docs` → Swagger UI (test in the browser)

```bash
.venv\Scripts\uvicorn src.api:app --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/auth/login -H "Content-Type: application/json" -d "{\"username\": \"demo_admin\", \"password\": \"132336BV132336\"}"
# → {"token": "...", "user_id": 1, "role": "Admin", ...}
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -H "Authorization: Bearer <token>" -d "{\"text\": \"ផលិតផលល្អណាស់\"}"
```

**API security (hardened)** — `src/common/security.py` + `slowapi`:

| Layer | Mechanism |
|---|---|
| Auth | HMAC-SHA256 signed tokens (stdlib, no JWT dep), 24 h expiry, secret from `API_SECRET` (in `.env`, never committed); exposed as an `HTTPBearer` security scheme → Swagger UI **Authorize** button |
| Rate limits | `5/min` `/auth/login`, `120/min` `/predict`, `30/min` `/feedback` per IP (env-tunable) |
| Brute force | 5 failed logins → 15-min lockout (in-memory) |
| Input caps | text ≤ 2000 chars, username ≤ 64, password ≤ 128 (reject 422) |
| Authorization | `GET /feedback` = Admin only (403 for `User`) |
| CORS | locked to `ALLOWED_ORIGINS` env (never `*`) |
| No PII | server logs no request bodies or tokens |
| HTTPS | non-localhost: `uvicorn --ssl-keyfile key.pem --ssl-certfile cert.pem` or reverse proxy |

Demo admin credentials (seeded in `src/db/schema.sql`): `demo_admin` / `132336BV132336`,
`demo_user` / `user@132123` — change them in production.

**Unseen-data API test** — `scripts/api_test_unseen.py` sends all 989 held-out
rows through the running API (logs in first, 8 workers, ~2–3 min) →
`reports/api_test_unseen.json`:

| Through | Accuracy | Macro-F1 |
|---|---|---|
| Offline eval (v2) | 0.8241 | 0.7543 |
| **Live API (v2, secured)** | **0.8241** | **0.7543** |

**Exact parity (0 errors)** — the deployed server serves the same v2 model,
no drift. 10 new API security tests (TestClient, mocked pipeline) →
**115/115 tests**.

### Swagger UI fix — Authorize button (done 2026-08-06)

**Problem:** auth was read manually with `authorization: str = Header(None)`.
Swagger UI silently drops header parameters named `authorization` — the field
you typed the token into was never sent, so every `/predict` / `/feedback`
call failed with `401 missing bearer token` even with a valid token.

**Fix:** `src/api.py` now uses the standard `HTTPBearer(auto_error=False)`
security scheme + a `_bearer_token` dependency (verify-token logic unchanged):

```python
_bearer = HTTPBearer(auto_error=False)

def _bearer_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    user = verify_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user
```

Result: Swagger shows a real **Authorize** button; curl behavior is identical
(`Authorization: Bearer <token>`). Verified end-to-end against the live API
(`demo_admin` login → real HMAC token → `/predict` = 200 with real model output,
e.g. mixed text → positive 0.833, Service aspect hit, anger emotion active).

### Step-by-step API test guide

Prereqs: uvicorn running (`.venv\Scripts\uvicorn src.api:app --reload --host 127.0.0.1 --port 8001`),
PostgreSQL seeded (`src/db/schema.sql` — `demo_admin` / `password_admin`, `demo_user` / `password_user`).

1. **Health (no auth):** `GET /health` → `{"status":"ok","model":"khmer-sentiment-3class-v2","db":true}`
2. **Login:** `POST /auth/login` with `{"username":"demo_admin","password":"132336BV132336"}`
   → `{"token":"...","role":"Admin","expires_in_seconds":86400}` (token valid 24 h)
3. **Authorize (once per Swagger session):** click **Authorize** → paste the token
   (no `Bearer`, no quotes) → Authorize → Close
4. **Predict:** `POST /predict` with `{"text":"សេវាកម្មនៅទីនេះ slow ខ្លាំងណាស់ ...","consent":false}`
   → `200`: language, sentiment, real confidence, aspects (business_aspects +
   emotions)
5. **Feedback (Admin only):** `GET /feedback` → `200` with rows (or `[]`)
6. **Negative tests (must fail = pass):**
   - no token → `401 missing bearer token`
   - `Authorization: Bearer garbage` → `401 invalid or expired token`
   - non-Admin token on `/feedback` → `403 admin role required`
   - text > 2000 chars → `422`
   - 6th fast `/auth/login` attempt → `429 rate limit exceeded`

PowerShell tip: inline JSON in `curl.exe` args gets mangled — write the body to a
file first: `--data-binary "@path\to\body.json"`.

---

## Calibration & robustness

`scripts/check_calibration.py` on val (1,877 rows) → `reports/calibration.json`:

| Confidence bin | Rows | Avg conf | Accuracy |
|---|---|---|---|
| 0.5–0.6 | 75 | 0.553 | 0.613 |
| 0.6–0.7 | 71 | 0.657 | **0.423** |
| 0.7–0.8 | 78 | 0.754 | 0.628 |
| 0.8–0.9 | 171 | 0.854 | 0.702 |
| 0.9–1.0 | 1,475 | 0.981 | 0.915 |

- **ECE = 0.083** — the model is overconfident in the middle range (0.6–0.7 bin: says 66% confident, only 42% right)
- Rows with confidence < 0.90 (21%) are unreliable; rows ≥ 0.90 are well calibrated (0.915 acc)
- **UI rule:** treat `confidence < 0.90` as "uncertain" → show a neutral/ask-user state instead of a guess

## Workflow improvements

- **Smoke modes**: `train_3class.py --smoke` (200 rows, 1 epoch, temp out dir — real model untouched), `--max-rows N` on eval scripts
- **Pre-flight gate**: `scripts/preflight_check.py` — compile + import check in seconds
- **Script tests**: `tests/test_scripts.py` (17 tests) — **caught a real bug** (`tykea_to_3probs()` didn't sum to 1; fixed + normalized)

---

## Structure

```
Update_Project_Internship/
|-- app/
|   |-- dashboard.py              # Streamlit entry (run this)
|   |-- app_pages/
|   |   |-- chat_agent.py         # Ask the AI agent page (AI settings panel)
|   |   |-- feedback.py           # Admin feedback page
|   |   |-- predict.py            # Analyze a comment page
|   |   `-- unseen_test.py        # Test data page (paste/CSV/989 benchmark)
|   |-- ai_agent.py               # offline explainer + local/cloud LLM tiers
|   |-- api_client.py             # pure HTTP client (token per request)
|   |-- dashboard_utils.py        # cached client + auth
|   `-- unseen_eval.py            # evaluation logic (no Streamlit)
|-- data/
|   |-- raw/                      # source files
|   |-- labeled/
|   |   `-- external_processed.csv     # 18,771 labeled rows
|   |-- splits/                   # train 15,016 / val 1,877 / test 1,878
|   |-- external_splits/          # train_mix 7,905 / val_mix 988 / test_ext 989 (held-out)
|   `-- external_kh_polarity.csv  # 9,882 unseen benchmark rows
|-- src/
|   |-- preprocessing/
|   |   |-- clean.py
|   |   |-- segment.py
|   |   |-- language_detect.py
|   |   `-- anonymize.py
|   |-- models/
|   |   |-- hf_api.py             # tykea live API + fallback
|   |   |-- local_model.py        # 3-class xlm-r predict / predict_fallback
|   |   |-- aspects.py            # 5 business aspects + 8 emotions
|   |   `-- translate_baseline.py
|   |-- common/
|   |   |-- config.py             # paths from config.yaml
|   |   |-- db.py                 # psycopg2
|   |   `-- security.py           # HMAC tokens + lockout
|   |-- db/
|   |   `-- schema.sql            # tables + triggers + bcrypt functions
|   |-- predict.py                # detect → 3-class local model (no translate)
|   `-- api.py                    # FastAPI: /health /auth/login /predict /feedback /docs
|-- models/
|   |-- khmer-sentiment-3class/       # v1 (for comparison)
|   |-- khmer-sentiment-3class-v2/    # v2 — PRODUCTION
|   |-- khmer-aspects-multilabel/     # songhieng cached
|   `-- stack_folds/
|-- scripts/
|   |-- prepare_splits.py
|   |-- preflight_check.py
|   |-- train_3class.py
|   |-- evaluate_baseline.py
|   |-- evaluate_final.py
|   |-- evaluate_external.py
|   |-- stack_phase3.py
|   |-- api_test_unseen.py
|   |-- check_calibration.py
|   |-- predict_demo.py
|   `-- compare_en_direct_vs_translate.py
|-- tests/                           # 233 tests (14 files)
|-- reports/                         # baseline/val/test/calibration/unseen JSONs
|-- docs/                            # consent.md · week4_log.md · week5_log.md · week6_log.md
|-- logs/
|-- .streamlit/                      # dashboard theme
|-- config.yaml                      # model paths, uncertainty_threshold 0.90
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml               # api + db
|-- pytest.ini
|-- .env  /  .env.example            # API_SECRET, DB creds, AI-provider keys (never commit .env)
`-- README.md
```

---


### Resources & references (models, data, tools)

Everything below is used directly in this project — base models, training
data, and libraries:

| Resource | What it is | Where we use it |
|---|---|---|
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) | Multilingual transformer (100 languages, incl. Khmer) — our **base model** | Fine-tuned → `models/khmer-sentiment-3class-v2` (the production sentiment model) |
| [5oni7a/Khmer-Profanity](https://huggingface.co/datasets/5oni7a/Khmer-Profanity) | ~18.7k labeled Khmer food-review comments — **main training data** | Cleaned → `data/labeled/external_processed.csv` (18,771 rows) |
| [ye-kyaw-thu/kh-polarity](https://github.com/ye-kyaw-thu/kh-polarity) | iSAI-NLP annotated Khmer polarity corpus (news/politics) — **unseen benchmark** | Mixed 7,905 rows into training (v2); 989 rows held out for external validation |
| [songhieng/khmer-xlmr-base-sentimental-multi-label](https://huggingface.co/songhieng/khmer-xlmr-base-sentimental-multi-label) | Khmer multi-label emotion model (8 emotions) | Cached at `models/khmer-aspects-multilabel/` → aspect analysis |
| [tykea/khmer-text-sentiment-analysis-roberta](https://huggingface.co/tykea/khmer-text-sentiment-analysis-roberta) | Khmer 2-class sentiment model (positive/negative) | Fallback path (`predict_fallback`) when the production model cannot load |
| [khmer-nltk](https://github.com/VietAI/khmer-nltk) | Khmer word segmentation + POS | `src/preprocessing/segment.py`; the AI agent's topic discovery |
| [Hugging Face transformers](https://github.com/huggingface/transformers) | Model training / inference framework | All model loading, fine-tuning (`train_3class.py`), evaluation |
| [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) | Python web API | `src/api.py` — the production prediction server |
| [Streamlit](https://streamlit.io/) | Python dashboard framework | `app/` — the user-facing dashboard |
| [PostgreSQL](https://www.postgresql.org/) | Database | Users, anonymized feedback, analysis records (`src/db/schema.sql`) |
| [Ollama](https://ollama.com) + [Qwen 2.5 3B](https://ollama.com/library/qwen2.5) | Local LLM for the chat assistant | Auto-detected local AI (no key/internet) in "Ask the AI agent" |

**Methodology references** — the design decisions in this project follow
established practice:

- Confidence-based abstention (`uncertain` when conf < 0.90) — the
  model never silently guesses; see the **Calibration & robustness**
  section (ECE 0.086, ≥0.90 bin = 0.91 acc).
- External / out-of-domain validation — the v1 → v2 retrain story
  (0.4014 → 0.8241 on unseen data) is documented in the **Week 5**
  section and `docs/week5_log.md`.
- Project repository: https://github.com/doeunbunheng/khmer_sentiment_engine
