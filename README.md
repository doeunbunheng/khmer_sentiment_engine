# Khmer Sentiment Engine

3-class sentiment (Positive / Negative / Neutral) + aspect analysis for
Khmer, English, and code-switched consumer comments. Roadmap: `docs/roadmap_publication.md`.

## Quick start

```bash
# 1. Environment
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Data splits (reads data/labeled/*.csv, writes data/splits/)
.venv\Scripts\python scripts\prepare_splits.py

# 3. Tests
.venv\Scripts\python -m pytest tests -v

# 4. Database (local PostgreSQL, or: docker compose up -d db)
psql -U postgres -h localhost -f src/db/schema.sql
```


## Progress by week

### Week 4 — 3-Class Fine-Tune + Stacking Ensemble

| Gate | Result |
|---|---|
| Phase 0 — Environment (DONE) | Fresh venv (Python 3.14.6), CUDA torch `2.13.0+cu130` on RTX 4060 (verified: matmul OK), `datasets` / `evaluate` added, old venv backed up to `.venv-backup` |
| Phase 1 — Baseline (DONE) | `scripts/evaluate_baseline.py` on `test.csv` → accuracy **0.6821**, macro-F1 **0.6071** (neg 0.73 / neu 0.33 / pos 0.76); by language: khmer 0.694 acc, english 0.458, mixed n=7 → `reports/baseline.json` |
| Phase 2 — Train 3-class (DONE) | `scripts/train_3class.py` — **xlm-roberta-base** → `models/khmer-sentiment-3class/`; neutral oversampled 2.5×, weighted loss, early stop on val macro-F1. **Val: accuracy 0.8503, macro-F1 0.8430** (neg 0.841 / neu 0.815 / pos 0.873); by language: km 0.855, en 0.766, code-switched 0.833 → `reports/phase2_val.json` |
| Phase 3 — Stacking (DONE, no gain) | `scripts/stack_phase3.py` — 5-fold OOF (xlm-r 3-class, 2 ep/fold) + tykea pseudo-probs + language one-hot → LogisticRegression meta. **Val acc 0.8450 / macro-F1 0.8370 — BELOW the Phase 2 single model (0.8503 / 0.8430)**. tykea features actively hurt (coef analysis + all variants); OOF xlm-r alone 0.8476/0.8413. Conclusion: stacking adds no value here → Phase 4 uses the Phase 2 single model. OOF features cached in `reports/phase3_oof.npz` |
| Phase 4 — Final test (DONE) | `scripts/evaluate_final.py` — one-shot on `test.csv` (1,878 rows) with `models/khmer-sentiment-3class`: **accuracy 0.8211, macro-F1 0.8156** (neg 0.809 / neu 0.796 / pos 0.842); by language: km 0.8259 acc · en 0.7188 · code-switched 1.0 (n=7) → `reports/phase4_test.json`. vs baseline +0.139 acc, +0.209 macro-F1 |
| Phase 5 — Integration (DONE) | `config.yaml` → `local_model_path: models/khmer-sentiment-3class`; `src/predict.py` now runs the 3-class model directly — **EN→KM translation and the 0.60 neutral-threshold rule removed** (model handles Khmer/English/mixed natively); `local_model.py` split into production (`predict`) vs tykea fallback (`predict_fallback`); tests rewritten → **53/53 passed** |
| Phase 5 — Integration | *next* — point `config.yaml` at new model, drop neutral-threshold rule, update tests |


### Week 1 — Setup & Data Prep

| Gate | Result |
|---|---|
| Environment | Python 3.14 venv, `requirements.txt`, `.env` / `config.yaml` |
| Cleaning | `clean.py`: URLs, emails, mentions, hashtags, Unicode NFC, whitespace |
| Segmentation | `segment.py`: khmer-nltk word segmentation (regex fallback) |
| Labeled data | `data/labeled/external_processed.csv` — 18,771 unique labeled rows |
| Split 80/10/10 (stratified) | `data/splits/` train 15,016 / val 1,877 / test 1,878 (CSV + `splits_report.json`) |

### Week 2 — Detection, Sentiment, Resilience

| Gate | Result |
|---|---|
| Language detection | `khmer` / `english` / `mixed` / `unknown` (Unicode regex) |
| Sentiment (tykea model) | Live: Khmer, English (auto EN→KM translate), mixed — all return (label, score) |
| Neutral rule | score < 0.60 → `neutral` (3rd class without retraining) |
| API resilience | HF `api-inference.huggingface.co` DNS-dead globally → automatic local fallback (same model, cached) |
| `pytest tests/ -v` | 35/35 passed |

### Week 3 — Postgres + Secure Feedback Loop

| Gate | Result |
|---|---|
| Login / Register | Secure: bcrypt hash (pgcrypto), SQL `register_user()` / `login_user()`, roles `Admin` / `User` |
| Anonymization | Emails, phones, Khmer IDs, URLs, mentions, names → `[ANONYMIZED]` before save; trigger-enforced |
| Consent gate | `consent_granted` required — no save without consent (Python + DB trigger) |
| Feedback loop | `predict_and_save()` → anonymized row persisted for every agreed prediction |
| Postgres | `khmer_sentiment` DB: `users`, `user_profiles`, `user_feedback`, `analysis_records` |
| `pytest tests/ -v` | 52/52 passed |

## Structure

```
src/preprocessing/clean.py        cleaning (URLs, mentions, unicode NFC, whitespace)
src/preprocessing/segment.py      khmer-nltk word segmentation (regex fallback)
src/preprocessing/language_detect.py  khmer/english/mixed detection
src/preprocessing/anonymize.py    PII stripping before save (URL / email / phone / ID / name)
src/common/config.py              paths + params from config.yaml
src/common/db.py                  psycopg2: connect / register_user / login_user / save_feedback / fetch_feedback
src/db/schema.sql                 PostgreSQL tables + triggers + functions + views (pgcrypto bcrypt)
src/models/hf_api.py              HF Inference API → local fallback chain
src/models/local_model.py         transformers inference — production 3-class (`predict`) + tykea fallback (`predict_fallback`)
src/models/translate_baseline.py  Google Translate EN→KM (baseline only, no longer in the production path)
src/models/aspects.py             songhieng multi-label model (lazy-load, Week 5)
src/predict.py                    detect → 3-class local model (no translate / no threshold rule)
scripts/prepare_splits.py         load → clean → dedupe → stratified split → CSV+JSON
docs/consent.md                   user consent text (Khmer + English)
tests/                            clean / segment / language / hf_api / predict / splits / anonymize / auth / db
```
## Completed status (Week 3)

| Gate | Result |
|---|---|
| Split 80/10/10 (stratified) | train 15,016 / val 1,877 / test 1,878 rows, from 18,771 unique labeled rows |
| `pytest tests/ -v` | 52/52 passed |
| Postgres | OK (`khmer_sentiment` DB: `users`, `user_profiles`, `user_feedback`, `analysis_records`) |
| Login / Register | Secure: bcrypt hash (pgcrypto), SQL `register_user()` / `login_user()`, roles `Admin` / `User` |
| Anonymization | Emails, phones, Khmer IDs, URLs, mentions, names → `[ANONYMIZED]` before save; enforced by trigger |
| Consent gate | `consent_granted` required — no save without consent (Python + DB trigger) |
| Feedback loop | `predict_and_save()` → anonymized row persisted for every agreed prediction |
| Language detection | `khmer` / `english` / `mixed` / `unknown` (Unicode regex) |
| Sentiment (tykea model) | Live: Khmer, English (auto EN→KM translate), mixed — all return (label, score) |
| Neutral rule | score < 0.60 → `neutral` (3rd class without retraining) |
| API resilience | HF serverless host (`api-inference.huggingface.co`) DNS-dead globally → automatic local fallback of the same tykea model |


## Next (Week 4, Phase 1)

Score the current pipeline (tykea + 0.60 neutral rule) on `data/splits/test.csv`
→ accuracy + macro-F1 + per-language breakdown → `reports/baseline.json`.
That number is the target the fine-tuned 3-class model must beat.


#### Completed phases — step by step

**Phase 0 — Environment (27 min)**

1. Backed up the old venv: `ren .venv .venv-backup` (still there — safe to delete once new venv is trusted).
2. Recreated a fresh venv: `py -3.14 -m venv .venv` (Python 3.14.6).
3. Installed CUDA torch `2.13.0+cu130` (RTX 4060, 8 GB) from the PyTorch cu130 index — the PyPI default is CPU-only.
4. Installed the rest of `requirements.txt` (torch line removed) + `datasets` + `evaluate` + `accelerate`.
5. Verified: `torch.cuda.is_available() → True`, GPU matmul OK, existing pipeline still predicts (tykea, same result).

**Phase 1 — Baseline (target to beat) 5mn 22s⚠️ mandatory**

1. Wrote `scripts/evaluate_baseline.py` — runs the *production fallback path*: language detect → EN→KM translate (best-effort) → tykea 2-class → `score < 0.60 → neutral`.
2. Batched local inference on GPU (skips the dead HF API → seconds instead of hours).
3. Ran on `data/splits/test.csv` (1,878 rows; 96 translated, 0 failures).
4. Results saved to `reports/baseline.json`:

| Metric | Value |
|---|---|
| Accuracy | **0.6821** |
| Macro-F1 | **0.6071** (neg 0.73 / neu 0.33 / pos 0.76) |
| By language | khmer 0.694 acc · english 0.458 acc · mixed n=7 |

Key finding: the 0.60-threshold rule cannot detect neutral (F1 0.33) and English is nearly random (0.458) → a real 3-class model is required.

**Phase 2 — Train the 3-class model (xlm-roberta-base) (11mn)**

1. Wrote `scripts/train_3class.py`:
   - Loads train (15,016) / val (1,877); labels `negative=0, neutral=1, positive=2`.
   - **Neutral oversampled 2.5×** (train only: 2,489 → 6,222; val untouched) + **class-weighted loss** (neutral weight 2.01).
   - `xlm-roberta-base`, 3 labels, seed 42, `max_length=256`, batch 16, lr 2e-5, warmup 350 steps, 3 epochs, fp16.
   - **Early stopping on val macro-F1** (patience 2); best checkpoint kept (`load_best_model_at_end`).
2. Ran on GPU: 3,516 steps in **10.5 minutes**.
3. Saved model → `models/khmer-sentiment-3class/` (config + tokenizer + 1.06 GB safetensors).
4. Val evaluation → `reports/phase2_val.json`:

| Metric | Value |
|---|---|
| Accuracy | **0.8503** |
| Macro-F1 | **0.8430** |
| Per-class F1 | negative 0.841 · neutral **0.815** · positive 0.873 |
| By language | km 0.855 acc · en 0.766 acc · code-switched 0.833 (n=6) |
| Confusion | 609/718 neg right · 275/311 neu right · 712/848 pos right |

Headline: neutral F1 jumped 0.33 → 0.815 and English acc 0.458 → 0.766, without touching the test set.

**Remaining:** Phase 3 verdict recorded (stacking no-gain) → Phase 4 final test done (acc 0.8211 / macro-F1 0.8156) → Phase 5 integration done (53/53 tests) → **Week 5: aspects** (`src/models/aspects.py`, lazy-load) + deployment.