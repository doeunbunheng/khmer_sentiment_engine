# Week 4 Log — 3-Class Fine-Tune + Integration

Detailed step-by-step record of everything done in Week 4. The README keeps
only the summary; this file is the full log. New phases/experiments append
here **and** add 2-3 lines to the README Changelog.

## Phase 0 — Environment (27 min)

1. Backed up the old venv: `ren .venv .venv-backup` (still there — safe to delete once new venv is trusted).
2. Recreated a fresh venv: `py -3.14 -m venv .venv` (Python 3.14.6).
3. Installed CUDA torch `2.13.0+cu130` (RTX 4060, 8 GB) from the PyTorch cu130 index — the PyPI default is CPU-only.
4. Installed the rest of `requirements.txt` (torch line removed) + `datasets` + `evaluate` + `accelerate`.
5. Verified: `torch.cuda.is_available() → True`, GPU matmul OK, existing pipeline still predicts (tykea, same result).

## Phase 1 — Baseline (target to beat) — 5m 22s

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

## Phase 2 — Train the 3-class model (xlm-roberta-base) — 11m

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

## Phase 3 — Stacking ensemble — DONE, no gain (51.6 min)

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
5. Note (post-hoc fix): `tykea_to_3probs()` was later fixed to normalize to a sum of 1 (found by `tests/test_scripts.py`). The stored OOF features used the unnormalized version; the "no gain" conclusion is unaffected since it held across all feature variants.

## Phase 4 — One-shot final test — DONE (a few minutes)

1. Wrote `scripts/evaluate_final.py` — loads `models/khmer-sentiment-3class`, batched GPU inference on the **untouched** `data/splits/test.csv` (1,878 rows).
2. Final published numbers → `reports/phase4_test.json`:

| Metric | Value |
|---|---|
| Accuracy | **0.8211** |
| Macro-F1 | **0.8156** (neg 0.809 / neu 0.796 / pos 0.842) |
| By language | km 0.8259 acc · en 0.7188 acc · code-switched 1.0 (n=7) |
| Confusion | 588/719 neg right · 265/311 neu right · 689/848 pos right |

vs baseline: **+0.139 accuracy, +0.209 macro-F1**.

## Phase 5 — Integration — DONE

1. `config.yaml`: added `local_model_path: models/khmer-sentiment-3class`; `local_max_length` 512 → 256 (matches training).
2. `src/common/config.py`: added `MODEL_DIR` constant.
3. `src/models/local_model.py`: split into production **`predict`** (3-class local model) and **`predict_fallback`** (tykea — keeps `evaluate_baseline.py` + HF API fallback working).
4. `src/models/hf_api.py`: fallback now calls `predict_fallback` (API model == fallback model).
5. `src/predict.py`: classifies directly with the 3-class model — **EN→KM translation and the 0.60 neutral-threshold rule removed** (model handles Khmer/English/mixed natively).
6. `tests/test_predict.py` rewritten for the new pipeline (8 tests: direct 3-class, no-translate English, low-confidence keeps label, failure → neutral, unknown label → neutral).
7. Verified live: Khmer pos 0.986 · Khmer neg 0.994 · English pos 0.990 · English neg 0.949 · mixed pos 0.817.
8. **`pytest tests/ -v` → 53/53 passed.**

## English experiment — direct vs EN→KM translation — DONE

Ran `scripts/compare_en_direct_vs_translate.py` on the 96 English `test.csv` rows with the same 3-class model:

| Metric | English DIRECT | English → Khmer (translated) |
|---|---|---|
| Accuracy | **0.7188** | 0.6875 |
| Macro-F1 | **0.6594** | 0.6359 |
| Speed | **1.3 ms/row** | 295 ms/row |
| Label flips vs direct | — | 13/96 rows (13.5%) changed |

Timing (measured, 96 rows): direct 0.085 s total (0.9 ms/row) vs translate+predict 27.84 s (290 ms/row) → **~329× faster**.

Decision: keep the no-translation pipeline. Overall test effect: 0.8211 (direct) vs 0.8195 (translate). Historical proof: old tykea+translate scored 0.458 on English — the model was the problem, not the language. → `reports/english_direct_vs_translate.json`.

## Full results — training / validation / testing

| Set | Rows | Accuracy | Macro-F1 |
|---|---|---|---|
| Training (5-fold OOF, honest*) | 15,016 | 0.8358 | 0.8272 |
| Validation (early-stopped) | 1,877 | 0.8503 | 0.8430 |
| Testing (untouched, final) | 1,878 | 0.8211 | 0.8156 |

Per-class F1:

| Set | negative | neutral | positive |
|---|---|---|---|
| Train | 0.8264 | 0.7950 | 0.8602 |
| Val | 0.8412 | 0.8148 | 0.8731 |
| Test | 0.8094 | 0.7958 | 0.8418 |

* OOF = each train row predicted by a fold model that never trained on it (no leakage). In-sample train accuracy (~0.97+) is meaningless; OOF is the honest number.

Sources: `reports/phase2_val.json`, `reports/phase4_test.json`, `reports/phase3_oof.npz`.

## Calibration check — DONE (post-Week-4)

Ran `scripts/check_calibration.py` on val (1,877 rows) → `reports/calibration.json`:

| Confidence bin | Rows | Avg conf | Accuracy |
|---|---|---|---|
| 0.5-0.6 | 75 | 0.553 | 0.613 |
| 0.6-0.7 | 71 | 0.657 | **0.423** |
| 0.7-0.8 | 78 | 0.754 | 0.628 |
| 0.8-0.9 | 171 | 0.854 | 0.702 |
| 0.9-1.0 | 1475 | 0.981 | 0.915 |

- ECE = 0.083; overall acc 0.8503.
- **Finding: the model is overconfident in the middle range** — the 0.6-0.7 bin says 66% confident but is only 42% right. Rows with confidence < 0.9 (395 rows, 21%) are unreliable (0.42-0.70 acc); rows ≥ 0.9 are well calibrated (0.915 acc).
- **Action for UI**: treat `confidence < 0.90` as "uncertain" → show a neutral/ask-user state instead of a guess. Only 21% of rows hit this, and they are exactly the risky ones.
- Optional next step if needed: temperature scaling to reduce ECE (not yet applied).

## Workflow improvements — DONE (post-Week-4)

1. **Smoke modes**: `train_3class.py --smoke` (200 rows, 1 epoch, temp out dir — never touches the real model), `evaluate_final.py --max-rows N`, `compare_en_direct_vs_translate.py --max-rows N`, `stack_phase3.py --folds/--fold-epochs/--max-train-rows`.
2. **Pre-flight gate**: `scripts/preflight_check.py` — py_compile all 24 files + import all 11 modules in seconds, exit 1 on failure.
3. **Script unit tests**: `tests/test_scripts.py` (17 tests) — `to_3class()` threshold mapping, `tykea_to_3probs()` math, report JSON schema. **Caught a real bug**: pseudo-probabilities didn't sum to 1 (now fixed + normalized).
