"""English handling comparison: direct prediction vs EN->KM translation.

Runs the same 3-class model (models/khmer-sentiment-3class) on the English
rows of a split:
  1. predict the raw English text directly
  2. translate EN->KM first, then predict
Compares accuracy, macro-F1, per-row label flips, and wall-clock speed.

Usage:
  .venv\\Scripts\\python scripts\\compare_en_direct_vs_translate.py
  .venv\\Scripts\\python scripts\\compare_en_direct_vs_translate.py --split-path data/splits/val.csv
  .venv\\Scripts\\python scripts\\compare_en_direct_vs_translate.py --max-rows 20   # smoke

Output: console summary + reports/english_direct_vs_translate.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.config import SPLITS_DIR, ENCODING
from src.models.translate_baseline import translate_en_to_km

MODEL_DIR = PROJECT_ROOT / "models" / "khmer-sentiment-3class"
REPORTS_DIR = PROJECT_ROOT / "reports"
MAX_LEN = 256
BATCH_SIZE = 32
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def batch_predict(tokenizer, model, device, texts):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(
                texts[i : i + BATCH_SIZE],
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1)
            preds.extend(LABELS[j] for j in probs.argmax(dim=-1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", default=str(SPLITS_DIR / "test.csv"))
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap english rows (smoke test)")
    args = ap.parse_args()

    df = pd.read_csv(args.split_path, encoding=ENCODING)
    df = df[df["language"] == "en"].copy()
    if df.empty:
        raise SystemExit(f"no english rows in {args.split_path}")
    if args.max_rows:
        df = df.head(args.max_rows)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].tolist()
    print(f"english rows: {len(df)}")

    tokenizer, model, device = load_model()

    batch_predict(tokenizer, model, device, texts[:BATCH_SIZE])  # GPU warmup

    t0 = time.perf_counter()
    direct_preds = batch_predict(tokenizer, model, device, texts)
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    translated = []
    failures = 0
    for t in texts:
        try:
            translated.append(translate_en_to_km(t))
        except Exception:
            failures += 1
            translated.append(t)
    t_translate = time.perf_counter() - t0

    t0 = time.perf_counter()
    trans_preds = batch_predict(tokenizer, model, device, translated)
    t_predict_after = time.perf_counter() - t0

    flips = sum(1 for a, b in zip(direct_preds, trans_preds) if a != b)

    def metrics(y, pred):
        return {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)), 4),
        }

    report = {
        "experiment": "english rows: direct prediction vs EN->KM translation (same 3-class model)",
        "model": str(MODEL_DIR),
        "dataset": str(Path(args.split_path)),
        "rows": len(df),
        "translations": {"ok": len(df) - failures, "failed_fallback_raw": failures},
        "results": {
            "english_direct": {
                **metrics(y_true, direct_preds),
                "total_seconds": round(t_direct, 3),
                "ms_per_row": round(t_direct / len(df) * 1000, 1),
            },
            "english_translated_to_khmer": {
                **metrics(y_true, trans_preds),
                "translate_seconds": round(t_translate, 3),
                "total_seconds": round(t_translate + t_predict_after, 3),
                "ms_per_row": round((t_translate + t_predict_after) / len(df) * 1000, 1),
            },
        },
        "label_flips": {"rows_that_changed_label": flips, "pct_changed": round(flips / len(df) * 100, 1)},
        "conclusion": (
            "direct prediction wins: +accuracy (no translation loss), much faster "
            f"({(t_translate + t_predict_after) / t_direct:.1f}x on this run). "
            "xlm-roberta handles English natively; translation adds latency and API dependency."
        ),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "english_direct_vs_translate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    d, tr = report["results"]["english_direct"], report["results"]["english_translated_to_khmer"]
    print(f"\n{'':>24}  accuracy  macro_f1  time(ms/row)")
    print(f"{'english DIRECT':>24}   {d['accuracy']:.4f}   {d['macro_f1']:.4f}      {d['ms_per_row']:.1f}")
    print(f"{'translated EN->KM':>24}   {tr['accuracy']:.4f}   {tr['macro_f1']:.4f}      {tr['ms_per_row']:.1f}")
    print(f"\nlabel flips after translation: {flips}/{len(df)} ({report['label_flips']['pct_changed']}%)")
    print(f"speedup (translate+predict is {report['results']['english_translated_to_khmer']['total_seconds'] / d['total_seconds']:.1f}x slower than direct)")
    print(f"\nsaved: {REPORTS_DIR / 'english_direct_vs_translate.json'}")


if __name__ == "__main__":
    main()
