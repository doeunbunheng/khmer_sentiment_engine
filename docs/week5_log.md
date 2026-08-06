# Week 5 — Aspect Analysis (Log)

Date: 2026-08-06

## Goal

After Week 4 (3-class sentiment), add "what is the comment about" — the 5
business aspects (Price / Service / Product Quality / Authenticity / Delivery)
shown in the product UI, plus an emotion layer.

## Design decisions (agreed before implementation)

1. **Business aspects ≠ emotions.** The songhieng model outputs 8 emotions;
   the UI wants 5 business topics. Emotions alone cannot answer "what is the
   comment about". Data check: `external_processed.csv` has **no aspect
   labels**, so fine-tuning an aspect model is impossible this week.
   → **Hybrid:** keyword rules for the 5 aspects + songhieng model for emotions.
2. **All languages:** songhieng is xlm-roberta-base (multilingual) — no
   language filter.
3. **Return shape:** `{scores: {8 probs}, active: [prob >= 0.5]}`.
4. **Local cache:** download once → `models/khmer-aspects-multilabel/`
   (`aspect_model.local_path`), offline afterwards.

## Implementation

- `src/models/aspects.py`
  - `detect_business_aspects(text)` — Khmer (substring) + English (word-token)
    keyword dictionaries, multi-label, returns matched keywords per aspect.
  - `predict_emotions(text)` — songhieng lazy-load (`lru_cache`), local-cache
    first, HF otherwise, sigmoid multi-label, active = prob ≥ 0.5, label-count
    guard (`config.num_labels` mismatch raises).
  - `predict(text)` — combined output; every failure degrades to empty dicts.
- `src/predict.py` — `predict_sentiment()` returns
  `aspects: {business_aspects, emotions}`; empty for blank text; try/except
  so sentiment never dies on aspect errors. `predict_and_save()` unchanged
  (already forwards to the `aspects` JSONB column).
- `config.yaml` — `aspect_model.local_path` + `emotion_threshold: 0.5`.
- `tests/conftest.py` — autouse fixture blocks the ~1.2 GB download at the
  transformers layer; the real `_load` logic still runs under test.
- `tests/test_aspects.py` — 30 tests: rules (each aspect, Khmer+EN),
  multi-label, no-match, threshold/active, fake-model emotions, label-count
  mismatch, failure degradation, `predict_sentiment` integration.
- `tests/test_db.py` — aspects JSONB round-trip test.

## Verification

- `preflight_check.py` → OK
- `pytest tests` → **101/101 passed** (was 70) — songhieng never downloads in tests
- Live (first run, downloads ~1.2 GB):
  - `"ផលិតផលល្អណាស់ គុណភាពល្អ តម្លៃថោកសមរម្យ\nThe delivery was late and the price was too high"` (mixed)
  - sentiment: positive 0.873
  - aspects: Price (តម្លៃ, ថោក, price), Product Quality (គុណភាព, ផលិតផល), Delivery (delivery)
  - emotions: Anger active (0.704)
- Cached to `models/khmer-aspects-multilabel/` — second run ~17 s, no download.

## Known limits / next steps

- Khmer matching is substring-based → rare false positives (e.g. `ដើម` in
  `ដើមឈើ` "tree trunk"). Acceptable v1; revisit with segmentation if needed.
- English emotion accuracy on the songhieng model is unverified (Khmer-trained
  base); validated live on mixed input only.
- Grow keyword dictionaries from real comments.
- Optional future: fine-tune an aspect classifier once labeled aspect data
  exists (~1-2k rows).

## v2 � mixed-domain retrain (2026-08-06)

### Diagnosis
- Old model OOD: 72% of kh-polarity rows conf>=0.9 with only 0.37 acc (confident wrong).
- Not truncation (mean 49 tokens), not thresholds. Overfit to food-review register.

### Data
- kh-polarity 80/10/10 stratified seed 42 -> data/external_splits/ (train_mix 7905 / val_mix 988 / test_ext 989 held out).
- v2 train = 15,016 + 7,905 = 22,921; val = 1,877 + 988 = 2,865. Same hyperparams (LR 2e-5, 3 epochs, early stop, fp16, seed 42).

### Results
| Benchmark | v1 | v2 |
|---|---|---|
| In-domain test 1,878 | 0.8211/0.8156 | 0.8333/0.8291 |
| Unseen held-out 989 | 0.4014/0.4109 | 0.8241/0.7543 |
| Full corpus 9,882* | 0.4165/0.4216 | 0.9171/0.8982 |

* informational only - contains v2 training rows.

### Decision
- v2 in production (config local_model_path -> models/khmer-sentiment-3class-v2).
- v1 kept for comparison. 101/101 tests pass with v2.

## API security hardening (2026-08-06)
- Layers: HMAC bearer tokens (API_SECRET), slowapi rate limits (login 5/min, predict 120/min), 5-fail->15-min lockout, text<=2000, Admin-only /feedback, CORS whitelist, no-PII logs, TLS docs.
- POST /auth/login added; api_test_unseen.py logs in first (demo_admin / BV132336 — the seeded admin password).
- 989-row parity through secured API: 0.8241 / 0.7543, 0 errors (unchanged). Tests 105 -> 115.
