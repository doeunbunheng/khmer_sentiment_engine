# Khmer Sentiment Engine

3-class sentiment (Positive / Negative / Neutral) + aspect analysis for
Khmer, English, and code-switched consumer comments.

**Status: Week 5 complete** — 3-class sentiment + business aspects + 8-emotion
analysis, **v2 (mixed-domain) model in production**: in-domain test
0.8333 / 0.8291, unseen kh-polarity held-out 0.8241 / 0.7543. **Local API
deployment done** (`src/api.py` + FastAPI, unseen 989 rows through the API:
**0.8241 — exact parity with offline**). Calibration checked (ECE 0.083),
**105/105 tests pass**.

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
PostgreSQL seeded (`src/db/schema.sql` — `demo_admin` / `132336BV132336`, `demo_user` / `user@132123`).

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
src/preprocessing/     clean.py · segment.py · language_detect.py · anonymize.py
src/common/            config.py (paths from config.yaml) · db.py (psycopg2)
src/db/schema.sql      PostgreSQL tables + triggers + functions (pgcrypto bcrypt)
src/models/            hf_api.py · local_model.py (predict / predict_fallback) ·
                       translate_baseline.py · aspects.py (rules + emotions)
src/predict.py         detect → 3-class local model (no translate / no threshold)
src/api.py             FastAPI server: GET /health · POST /auth/login ·
                       POST /predict · GET /feedback · GET /docs
src/common/security.py HMAC-signed tokens + login lockout (API_SECRET from .env)
scripts/               prepare_splits · evaluate_baseline · train_3class ·
                       stack_phase3 · evaluate_final · evaluate_external ·
                       api_test_unseen · check_calibration ·
                       compare_en_direct_vs_translate · preflight_check · predict_demo
data/                  labeled/ (external_processed.csv) · splits/ ·
                       external_kh_polarity.csv (unseen benchmark) · external_splits/
docs/                  consent.md · week4_log.md · week5_log.md
tests/                 115 tests (app + scripts + aspects + API security)
```

---

## Next — Week 5 remainder

1. ~~**Aspects**~~ done: keyword business aspects + songhieng emotions, wired into `predict.py`
2. ~~**External validation**~~ done: kh-polarity unseen benchmark (0.4165 acc — domain shift finding)
3. ~~**v2 mixed retrain**~~ done: in-domain 0.8333 + unseen 0.8241, in production
4. ~~**Local API deployment**~~ done: FastAPI server + unseen API test (exact parity 0.8241)
5. ~~**API security hardening**~~ done: tokens, rate limits, lockout, admin-only feedback, CORS, no-PII logs — parity kept, 115/115
6. ~~**Docker**~~ done: Dockerfile + docker-compose (`api` + `db`), built and verified — health/login/predict through the container = same 0.8241 parity
7. ~~**Polish: grow aspect keyword dictionaries**~~ done: +30 keywords mined from the real corpus (data-backed, +24 tests)
8. ~~**OOD guard**~~ done: `uncertain` flag on every prediction when confidence < 0.90 (`config.yaml` `uncertainty_threshold`) — UI/ask-user state
9. ~~**Re-check calibration on v2**~~ done: ECE 0.0863 (val) / 0.1127 (unseen held-out); ≥0.90 bin = 0.91/0.89 acc — threshold 0.90 confirmed
10. **Frontend only**: consume `uncertain` in the UI (show neutral/ask-user state instead of a guess)

## Changelog

- **2026-08-06 (Docker + polish + OOD guard):** Docker verified — `docker-compose.yml` (`api` + `db`) built and run; health/login/predict through the container confirmed. Aspect dictionaries grown from the real corpus (co-occurrence mining on 18,771 comments): Price +បង់/ប្រូម៉ូសិន/pay/deal/sale, Service +អតិថិជន/អ្នកលក់/ម្ចាស់ហាង/seller/owner/customer, Quality +ប្រើប្រាស់/ខូច/នាំចូល/រឹងមាំ/broken/defective, Authenticity +ចម្លង/ត្រាប់/knockoff/dupe, Delivery +បញ្ជា/មកដល់/order/tracking (+24 tests). OOD guard: `predict_sentiment` now returns `uncertain` (conf < `model.uncertainty_threshold` 0.90), exposed in `PredictResponse` schema (+5 tests). Calibration re-checked on v2: val ECE 0.0863, unseen held-out ECE 0.1127; ≥0.90 bin = 0.9095 / 0.8911 acc — threshold confirmed; `check_calibration.py` now uses config model dir (`--model-dir`, `--report-name`). **144/144 tests pass.**

- **2026-08-06 (Swagger UI auth fix):** `src/api.py` switched from manual `Header(None)` parsing to the `HTTPBearer(auto_error=False)` security scheme with a `_bearer_token` dependency (HMAC verify logic unchanged, same 401 details). Swagger UI now shows a real **Authorize** button — previously header params named `authorization` were silently dropped by Swagger UI, causing `401 missing bearer token` on every `/predict` / `/feedback` call even with valid tokens. Verified live: login → token → `/predict` = 200 (mixed text → positive 0.833, Service aspect, anger emotion). Step-by-step test guide added to README.

- **2026-08-06 (API security hardening):** `src/api.py` hardened — HMAC-SHA256 bearer tokens (`src/common/security.py`, 24 h expiry, `API_SECRET`), slowapi rate limits (login 5/min, predict 120/min, feedback 30/min), 5-fail → 15-min login lockout, input caps (text ≤ 2000), Admin-only `GET /feedback`, CORS locked to `ALLOWED_ORIGINS`, no-PII logging, TLS docs. `POST /auth/login` added. Full 989-row parity re-run through the secured API = **0.8241 / 0.7543, 0 errors** (same as offline). 10 new security tests → **115/115 pass**. Demo admin password found in seed: `demo_admin` / `BV132336` (`src/db/schema.sql`).

- **2026-08-06 (Local API deployment):** `src/api.py` FastAPI (`/health`, `/predict`, `/docs`) serving the exact `predict_and_save()` pipeline, lazy model load. `scripts/api_test_unseen.py` sent all 989 held-out rows through the live API → **0.8241 / 0.7543 — exact parity with offline** (0 errors). 4 new API tests → **105/105 pass**. `fastapi` + `uvicorn` added to `requirements.txt`.

- **2026-08-06 (v2 mixed-domain retrain):** kh-polarity split 80/10/10 (train_mix 7,905 / val_mix 988 / test_ext 989 held-out). Trained v2 = 22,921 train rows, same hyperparams → **in-domain 0.8211 → 0.8333, unseen held-out 0.4014 → 0.8241** acc. v2 in production (`config.yaml`), v1 kept. `train_3class.py` gained `--extra-train/--extra-val/--out-dir/--report-name`; eval scripts gained `--model-dir/--dataset`.

- **2026-08-06 (Week 5 aspects + external validation):** `aspects.py` = 5 business aspects (Khmer+EN keyword rules, multi-label, matched keywords) + songhieng 8-emotion model (lazy-load, local cache `models/khmer-aspects-multilabel`, 0.5 threshold, label-count guard). Wired into `predict_sentiment` (failure-safe), stored via existing JSONB. 31 new mock-based tests → **101/101 pass**. Live verified Khmer+EN. External benchmark: kh-polarity (9,882 unseen rows, 0% overlap) → **0.4165 acc / 0.4216 macro-F1** — documented domain-shift limitation (`evaluate_external.py`, `data/external_kh_polarity.csv`).

- **2026-08-05 (Week 4 + hardening):** Phases 0–5 complete. Baseline 0.6821 → test **0.8211 / 0.8156**. Stacking no-gain. EN direct wins (accuracy + ~329× speed). Calibration ECE 0.083, uncertain < 0.90. Smoke modes + preflight + 17 script tests. **70/70 pass.**
- **2026-08-05 (Week 1–3):** Setup/data prep, detection/sentiment/resilience, Postgres + secure feedback loop.
- **2026-08-05 (repo init):** First commit pushed to GitHub (secret leak in `.env.example` fixed before push).
