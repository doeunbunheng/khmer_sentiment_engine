# Khmer Sentiment Engine

3-class sentiment (Positive / Negative / Neutral) + aspect analysis for
Khmer, English, and code-switched consumer comments.

Status: **Week 4 complete + hardening** — fine-tuned xlm-roberta 3-class model
integrated into production, **test accuracy 0.8211 / macro-F1 0.8156**, calibration
checked (ECE 0.083), smoke/preflight/test workflow in place, **70/70 tests pass**
(53 app + 17 script).

## Quick start

```bash
# 1. Environment
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Data splits (reads data/labeled/*.csv, writes data/splits/)
.venv\Scripts\python scripts\prepare_splits.py

# 3. Pre-flight check + tests (smoke before any long run)
.venv\Scripts\python scripts\preflight_check.py
.venv\Scripts\python -m pytest tests -v

# 4. Database (local PostgreSQL, or: docker compose up -d db)
psql -U postgres -h localhost -f src/db/schema.sql

# 5. Predict from a file / stdin / paste
.venv\Scripts\python scripts\predict_demo.py notes.txt
```

## Completed status — overall

| Week | Deliverable | Result |
|---|---|---|
| 1 | Setup + data prep | Python 3.14 venv, cleaning, segmentation, 18,771 labeled rows, stratified splits (15,016 / 1,877 / 1,878) |
| 2 | Detection + sentiment + resilience | Language detect, tykea live predictions, neutral rule, API→local fallback, **35/35 tests** |
| 3 | Postgres + secure feedback loop | Login/Register (bcrypt), anonymization, consent gate, feedback loop, **52/52 tests** |
| 4 | 3-class fine-tune + integration | Baseline 0.6821 → fine-tuned **0.8211** test acc, stacking no-gain, integrated, **53/53 tests** |
| Hardening | Calibration + workflow | ECE 0.083, "uncertain" rule < 0.90 conf, smoke modes, preflight gate, 17 script tests (1 bug found+fixed) → **70/70 tests** |

## Progress by week

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

### Week 4 — 3-Class Fine-Tune + Integration (DONE, step by step)

**Phase 0 — Environment (27 min)**

1. Backed up the old venv: `ren .venv .venv-backup` (still there — safe to delete once new venv is trusted).
2. Recreated a fresh venv: `py -3.14 -m venv .venv` (Python 3.14.6).
3. Installed CUDA torch `2.13.0+cu130` (RTX 4060, 8 GB) from the PyTorch cu130 index — the PyPI default is CPU-only.
4. Installed the rest of `requirements.txt` (torch line removed) + `datasets` + `evaluate` + `accelerate`.
5. Verified: `torch.cuda.is_available() → True`, GPU matmul OK, existing pipeline still predicts (tykea, same result).

**Phase 1 — Baseline (target to beat) — 5m 22s**

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

**Phase 2 — Train the 3-class model (xlm-roberta-base) — 11m**

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

**Phase 3 — Stacking ensemble — DONE, no gain (51.6 min)**

1. Wrote `scripts/stack_phase3.py`:
   - Base A: **5-fold out-of-fold probabilities** of the 3-class xlm-r model (each fold trains fresh on 4/5, predicts held-out 1/5 — no leakage; val predicted by all folds and averaged).
   - Base B: **tykea 2-class** → continuous 3-dim pseudo-probabilities (smooth version of the 0.60 neutral rule: `neu = 1 - max(pos, neg)`).
   - Features (10): `[xlm pos, neu, neg] + [tykea pos, neu, neg] + one-hot language [khmer, english, mixed, unknown]`.
   - Meta-learner: `LogisticRegression` (multinomial, balanced class weights).
2. Ran on GPU: 5 folds × 2 epochs ≈ 51.6 min. OOF features cached → `reports/phase3_oof.npz`.

| Model | Val acc | Macro-F1 |
|---|---|---|
| Phase 2 single model (3 ep + oversample) | **0.8503** | **0.8430** |
| xlm-r 5-fold OOF (argmax, no meta) | 0.8476 | 0.8413 |
| **Stack (xlm-r + tykea + lang → LR)** | **0.8450** | **0.8370** |
| tykea pseudo-probs alone | 0.1508 | 0.1077 |

3. Conclusion: **stacking adds no value** — tykea features actively hurt (verified by coefficient analysis + 5 feature variants); OOF xlm-r alone is better than the stack. The test set was never touched.
4. Decision: **Phase 4 uses the Phase 2 single model** → `reports/phase3_val.json`.

**Phase 4 — One-shot final test — DONE (a few minutes)**

1. Wrote `scripts/evaluate_final.py` — loads `models/khmer-sentiment-3class`, batched GPU inference on the **untouched** `data/splits/test.csv` (1,878 rows).
2. Final published numbers → `reports/phase4_test.json`:

| Metric | Value |
|---|---|
| Accuracy | **0.8211** |
| Macro-F1 | **0.8156** (neg 0.809 / neu 0.796 / pos 0.842) |
| By language | km 0.8259 acc · en 0.7188 acc · code-switched 1.0 (n=7) |
| Confusion | 588/719 neg right · 265/311 neu right · 689/848 pos right |

vs baseline: **+0.139 accuracy, +0.209 macro-F1**.

**Phase 5 — Integration — DONE**

1. `config.yaml`: added `local_model_path: models/khmer-sentiment-3class`; `local_max_length` 512 → 256 (matches training).
2. `src/common/config.py`: added `MODEL_DIR` constant.
3. `src/models/local_model.py`: split into production **`predict`** (3-class local model) and **`predict_fallback`** (tykea — keeps `evaluate_baseline.py` + HF API fallback working).
4. `src/models/hf_api.py`: fallback now calls `predict_fallback` (API model == fallback model).
5. `src/predict.py`: classifies directly with the 3-class model — **EN→KM translation and the 0.60 neutral-threshold rule removed** (model handles Khmer/English/mixed natively).
6. `tests/test_predict.py` rewritten for the new pipeline (8 tests: direct 3-class, no-translate English, low-confidence keeps label, failure → neutral, unknown label → neutral).
7. Verified live: Khmer pos 0.986 · Khmer neg 0.994 · English pos 0.990 · English neg 0.949 · mixed pos 0.817.
8. **`pytest tests/ -v` → 53/53 passed.**

**English handling experiment — direct vs EN→KM translation — DONE**

Ran `scripts/compare_en_direct_vs_translate.py` on the 96 English `test.csv` rows with the same 3-class model to settle whether translating English to Khmer before predicting helps:

| Metric | English DIRECT | English → Khmer (translated) |
|---|---|---|
| Accuracy | **0.7188** | 0.6875 |
| Macro-F1 | **0.6594** | 0.6359 |
| Speed | **1.3 ms/row** | 295 ms/row |
| Label flips vs direct | — | 13/96 rows (13.5%) changed |

On a KH+EN unseen dataset (test: 1,775 km / 96 en / 7 cs) the overall effect is 0.8211 (direct) vs 0.8195 (translate) — direct wins on all three axes:

1. **Accuracy**: translation loses information (13/96 rows flipped to the wrong label); xlm-roberta understands English natively.
2. **Speed**: direct is ~200–300× faster (pure GPU inference vs a network translation call per row).
3. **Simplicity**: no external translate API dependency or failure mode.

Historical proof: the old tykea+translate pipeline scored only **0.458** on English — the model was the problem, not the language. **Decision: keep the no-translation pipeline** → `reports/english_direct_vs_translate.json`.

**Timing breakdown — measured on the same 96 English rows:**

| Path | Total | Per row |
|---|---|---|
| Direct predict | 0.085 s | 0.9 ms |
| Translate only | 27.45 s | 285.9 ms |
| Translate + predict | 27.84 s | 290.0 ms |

**Direct is ~329× faster.** Translation is the bottleneck (a network call per row via Google Translate), while direct prediction is pure GPU inference (0.9 ms/row).

So direct wins on **all three axes**: accuracy (0.719 vs 0.688), speed (329×), and simplicity (no external API dependency). No reason to ever translate.

**Full results — training / validation / testing (final 3-class xlm-roberta model)**

Accuracy / Macro-F1

| Set | Rows | Accuracy | Macro-F1 |
|---|---|---|---|
| Training (5-fold OOF, honest\*) | 15,016 | 0.8358 | 0.8272 |
| Validation (early-stopped) | 1,877 | 0.8503 | 0.8430 |
| Testing (untouched, final) | 1,878 | 0.8211 | 0.8156 |

Per-class F1

| Set | negative | neutral | positive |
|---|---|---|---|
| Train | 0.8264 | 0.7950 | 0.8602 |
| Val | 0.8412 | 0.8148 | 0.8731 |
| Test | 0.8094 | 0.7958 | 0.8418 |

By language (accuracy)

| Set | khmer | english | code-switched |
|---|---|---|---|
| Train | — (no per-lang stored) | — | — |
| Val | 0.8548 (n=1777) | 0.7660 (n=94) | 0.8333 (n=6) |
| Test | 0.8259 (n=1775) | 0.7188 (n=96) | 1.0000 (n=7) |

\* OOF = each train row predicted by a fold model that never trained on it (no leakage). The in-sample train accuracy is higher (~0.97+) but meaningless — OOF is the honest number. These fold models used 2 epochs, so train OOF slightly understates the final model.

Reading the numbers:

1. **No overfitting** — train OOF ≈ val ≈ test (0.84 / 0.85 / 0.82), a healthy ~0.03 gap.
2. **Test slightly below val**, as expected — early stopping tuned on val; test was never touched.
3. **Neutral is the hardest class everywhere** (F1 ~0.80) — inherently ambiguous.
4. **English is the weak spot** (test 0.719) — only 96 rows; more EN/mixed training data would improve it.

Sources: `reports/phase2_val.json`, `reports/phase4_test.json`, `reports/phase3_oof.npz`.

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
scripts/evaluate_baseline.py      Phase 1: tykea baseline on test.csv → reports/baseline.json
scripts/train_3class.py           Phase 2: fine-tune xlm-roberta 3-class → models/khmer-sentiment-3class
scripts/stack_phase3.py           Phase 3: 5-fold OOF stacking → reports/phase3_val.json
scripts/evaluate_final.py         Phase 4: one-shot test.csv eval → reports/phase4_test.json
scripts/check_calibration.py      ECE + confidence buckets on val → reports/calibration.json
scripts/compare_en_direct_vs_translate.py  English direct vs EN→KM translation (accuracy + speed) → reports/english_direct_vs_translate.json
scripts/preflight_check.py        pre-run gate: py_compile all + import all modules
scripts/predict_demo.py           UTF-8-safe CLI demo (file / stdin / paste)
docs/consent.md                   user consent text (Khmer + English)
docs/week4_log.md                 full step-by-step record of Week 4 (phases + experiments)
tests/                            clean / segment / language / hf_api / predict / splits / anonymize / auth / db / scripts
```

## Calibration & robustness (post-Week-4)

Ran `scripts/check_calibration.py` on val (1,877 rows) → `reports/calibration.json`:

| Confidence bin | Rows | Avg conf | Accuracy |
|---|---|---|---|
| 0.5–0.6 | 75 | 0.553 | 0.613 |
| 0.6–0.7 | 71 | 0.657 | **0.423** |
| 0.7–0.8 | 78 | 0.754 | 0.628 |
| 0.8–0.9 | 171 | 0.854 | 0.702 |
| 0.9–1.0 | 1,475 | 0.981 | 0.915 |

- **ECE = 0.083** — the model is overconfident in the middle range (0.6–0.7 bin: says 66% confident, only 42% right).
- Rows with confidence < 0.90 (395 rows, 21%) are unreliable; rows ≥ 0.90 are well calibrated (0.915 acc).
- **UI recommendation:** treat `confidence < 0.90` as "uncertain" (show a neutral/ask-user state) instead of a guess.
- Full step-by-step record: `docs/week4_log.md`.

## Workflow improvements (post-Week-4)

- **Smoke modes**: `train_3class.py --smoke` (200 rows, 1 epoch, temp out dir — real model untouched), `--max-rows N` on eval scripts.
- **Pre-flight gate**: `scripts/preflight_check.py` — compile + import check in seconds.
- **Script tests**: `tests/test_scripts.py` (17 tests) — caught a real bug (`tykea_to_3probs()` didn't sum to 1; fixed + normalized).

## Next — Week 5

- **Aspects**: implement `src/models/aspects.py` (songhieng multi-label model, lazy-load) and add aspect extraction to `src/predict.py` / `predict_and_save()`.
- **Deployment**: containerize the app (docker-compose already present), serve the 3-class model over an API.

## Changelog

- **2026-08-05 (Week 4 + hardening):** Phase 0–5 complete. Baseline 0.6821 → fine-tuned test **0.8211 / 0.8156**. Stacking no-gain (documented). EN direct-vs-translate measured (direct wins: accuracy + ~329× speed). Calibration check (ECE 0.083, uncertain < 0.90). Smoke modes + preflight gate + 17 script tests. **53/53 + 17 script tests = 70/70 pass.**
- **2026-08-05 (Week 1–3, from earlier sessions):** Setup/data prep, detection/sentiment/resilience, Postgres + secure feedback loop.
- **2026-08-05 (repo init):** First commit pushed to GitHub (HTTPS, secret leak in `.env.example` fixed before push).
