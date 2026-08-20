# Khmer Sentiment Engine — Project Structure & What Was Completed

A machine-learning system that reads Khmer, English, and mixed (code-switched)
customer comments and tells a business: sentiment (positive / negative /
neutral), what it is about (5 business aspects + 8 emotions), an honest
"not sure" flag when confidence is low, and an AI chat assistant that explains
the results in plain language.

---

## 1. Project structure

```
D:\Update_Project_Internship
├── README.md                  # full documentation (executive summary + supervisor reference)
├── AGENTS.md                  # dev notes (Windows/Khmer UTF-8 quirks)
├── config.yaml                # data / model / aspect / db / thresholds config
├── requirements.txt           # pinned Python dependencies (19 packages)
├── pytest.ini                 # test config
├── .env                       # secrets (gitignored): DB, API_SECRET, YOUTUBE_API_KEY, ...
├── .env.example               # documented template of all env keys
├── Dockerfile                 # container for the FastAPI server
├── docker-compose.yml         # API + PostgreSQL compose
│
├── src/                       # core engine (importable package)
│   ├── api.py                 # FastAPI server: /auth, /predict, /feedback, rate limits
│   ├── predict.py             # main predict_sentiment() pipeline
│   ├── common/
│   │   ├── config.py          # config.yaml loader
│   │   ├── security.py        # HMAC tokens, login lockout, hashing
│   │   ├── db.py              # PostgreSQL access
│   │   └── collect.py         # shared CSV writer + row parsers for collectors
│   ├── collectors/            # data collection (NEW - Week 7)
│   │   └── youtube.py         # YouTube Data API: link→ID, video info, comments, quota budget
│   ├── models/
│   │   ├── local_model.py     # xlm-roberta 3-class v2 (production)
│   │   ├── aspects.py         # aspect + emotion multi-label model
│   │   ├── hf_api.py          # cloud fallback (Hugging Face)
│   │   └── translate_baseline.py  # old translate+classify baseline (kept for comparison)
│   ├── preprocessing/
│   │   ├── clean.py           # Khmer text cleaning
│   │   ├── segment.py         # Khmer word segmentation
│   │   ├── language_detect.py # khmer / english / mixed detection
│   │   └── anonymize.py       # PII removal for feedback storage
│   └── db/schema.sql          # users, feedback tables + demo admin seed
│
├── app/                       # Streamlit dashboard (user-facing)
│   ├── dashboard.py           # entry point: login + navigation
│   ├── api_client.py          # HTTP client for the FastAPI server
│   ├── ai_agent.py            # AI chat explainer logic (offline / local / Gemini / GPT)
│   ├── unseen_eval.py         # dataset eval logic (pure, unit-tested)
│   ├── dashboard_utils.py     # session helpers
│   └── app_pages/
│       ├── predict.py             # "Analyze a comment" (single prediction)
│       ├── unseen_test.py         # "Test data" (upload / paste / built-in datasets)
│       ├── youtube_analyzer.py    # "Analyze YouTube comments" (NEW - Week 7)
│       ├── chat_agent.py          # "Ask the AI agent" (discusses last result)
│       └── feedback.py            # "Feedback" (admin only)
│
├── scripts/                   # one-off / CLI tools
│   ├── prepare_splits.py          # labeled CSV → train/val/test (80/10/10 stratified)
│   ├── train_3class.py            # fine-tune xlm-roberta 3-class
│   ├── evaluate_final.py          # final model metrics on splits
│   ├── evaluate_external.py       # kh-polarity benchmark eval
│   ├── evaluate_baseline.py       # old baseline comparison
│   ├── stack_phase3.py            # stacking ensemble experiment (no gain, documented)
│   ├── check_calibration.py       # confidence calibration check
│   ├── api_test_unseen.py         # send rows through live API
│   ├── compare_en_direct_vs_translate.py
│   ├── predict_demo.py            # predict from file / stdin (safe Khmer input)
│   ├── preflight_check.py         # environment pre-flight gate
│   ├── collect_youtube_comments.py    # CLI: search topics → comments → CSV (NEW)
│   └── collect_maps_reviews.py        # CLI: places → reviews → CSV (NEW)
│
├── data/
│   ├── labeled/external_processed.csv   # 18,771 labeled Khmer comments
│   ├── splits/                          # train.csv 15,016 · val.csv 1,877 · test.csv 1,878
│   ├── external_kh_polarity.csv         # 9,882 news/politics benchmark
│   ├── external_splits/                 # train_mix / val_mix / test_ext (989 held-out)
│   └── raw/                             # collected data (gitignored)
│       ├── youtube_comments.csv         # 537 comments collected this week
│       └── youtube_<video_id>.csv       # per-video collections from the dashboard
│
├── models/                    # trained models (gitignored)
│   ├── khmer-sentiment-3class/      # first fine-tuned model (0.8211)
│   ├── khmer-sentiment-3class-v2/   # production: mixed-domain (0.8333 in-domain, 0.8241 unseen)
│   └── khmer-aspects-multilabel/    # aspects + emotions
│
├── tests/                     # 243 tests, all passing
│   ├── test_collectors.py     # (NEW) collectors: CSV schema, UTF-8, dedupe, URL parsing
│   └── test_*.py              # api, auth, predict, aspects, clean, segment, splits,
│                              # dashboard, ai_agent, db, language, anonymize, hf_api, scripts
│
├── reports/                   # JSON evaluation results per experiment
├── docs/                      # week logs + consent policy
└── logs/                      # api / streamlit runtime logs
```

---

## 2. What was completed — step by step

### Weeks 1–6 (before this week)
| Week | Completed |
|---|---|
| 1 | Collected & cleaned **18,771 labeled Khmer comments**; stratified train/val/test splits |
| 2 | Sentiment detection (Khmer/English/mixed) with automatic cloud fallback — 35/35 tests |
| 3 | PostgreSQL + secure login/register + anonymized feedback with consent — 52/52 tests |
| 4 | Fine-tuned own 3-class model (`xlm-roberta-base`, on this PC's GPU): 0.6821 → **0.8211**; honestly documented that a stacking ensemble gave no gain — 53/53 tests |
| 5 | Aspect analysis (5 aspects + 8 emotions); tested on truly unseen news/politics data → found real weakness (0.4165) → retrained mixed-domain **v2 model (0.8241 on unseen)**; FastAPI server + security + Docker — 144/144 tests |
| 6 | Complete Streamlit dashboard: login, live analysis, test your own data, **AI chat agent** (offline / local PC AI / one-click Gemini-GPT), admin feedback — 178/178 tests |
| 6b | AI agent upgraded to work on **any dataset** (discovers topics itself, provider switching) — **233/233 tests** |

### Week 7 — this session (all verified working)
| # | Completed | Files | Verified |
|---|---|---|---|
| 1 | **YouTube comment collector** — free official YouTube Data API v3, quota budget guard, dedupe, UTF-8 CSV | `src/collectors/youtube.py`, `scripts/collect_youtube_comments.py`, `src/common/collect.py` | Live: 537 comments (323 Khmer), 418 units used |
| 2 | **Google Maps reviews collector** (Places API) — ready, needs billing enabled to run | `scripts/collect_maps_reviews.py` | Ready (skipped — free-only decision) |
| 3 | **Dashboard page "Analyze YouTube comments"** — paste link → collect comments → run engine → charts, per-comment table, downloads | `app/app_pages/youtube_analyzer.py` | Live: 100 comments → 35/31/34 distribution, 0 errors |
| 4 | **Linked to the AI agent** — YouTube result auto-loads as the dataset the agent discusses, with an auto-typed first question | `youtube_analyzer.py` + existing `chat_agent.py` (fixed `type: "dataset"` context) | Live: agent answered correctly |
| 5 | **Env keys** — `YOUTUBE_API_KEY` + `GOOGLE_PLACES_API_KEY` slots | `.env`, `.env.example` | Key activated (2 APIs restricted) |
| 6 | **Tests** — 10 new tests (CSV schema, UTF-8 Khmer, dedupe, URL parsing, quota budget) | `tests/test_collectors.py` | |

---

## 3. Key numbers

| Metric | Value |
|---|---|
| Final model | `xlm-roberta-base` 3-class v2 |
| In-domain accuracy (1,878 rows) | **0.8333** |
| Unseen data accuracy (989 news/politics rows) | **0.8241** (fixed real domain-shift: old 0.4014) |
| Live API vs offline parity | 989/989 rows identical |
| Training data | 18,771 labeled Khmer comments + 9,882 benchmark |
| New collected data (this week) | 537 YouTube comments (60% Khmer) |
| YouTube collection cost | 100% free — 10,000 units/day (≈50,000 comments/day) |

---

## 4. How to run

```bash
# 1. API server (already running)
docker compose up -d or
.venv\Scripts\uvicorn src.api:app --host 127.0.0.1 --port 8000

# 2. Dashboard
.venv\Scripts\streamlit run app\dashboard.py

# 3. Log in: demo_admin / password

# 4. Collect YouTube comments from the CLI (free)
.venv\Scripts\python scripts\collect_youtube_comments.py --query "khmer news" --max-videos 5 --max-comments 200 --budget 2000

# 5. Or inside the dashboard: "Analyze YouTube comments" → paste link → Collect & analyze
```

---

## 5. Free-only data plan (no billing, no credit card)

| Source | Free | Card | Status |
|---|---|---|---|
| YouTube comments (Data API) | Yes, forever | No | ✅ Working |
| Telegram channels (telethon) | Yes | No | Not built yet (optional) |
| Facebook comments | Trail | No | just test for some extension |
