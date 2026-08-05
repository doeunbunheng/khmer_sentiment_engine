"""Phase 1 — Baseline evaluation of the current production pipeline on test.csv.

Pipeline under test (local model = production's fallback path, same tykea weights):
  detect language -> (EN->KM translate for english rows) -> tykea 2-class
  -> confidence < neutral_threshold => neutral  -> 3-class label

Batched inference on GPU for speed. Translation is best-effort (fails fall back
to raw text). Output reported to console and saved to reports/baseline.json.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.common.config import MODEL, SPLITS_DIR, ENCODING
from src.models.local_model import _load
from src.models.translate_baseline import translate_en_to_km
from src.preprocessing.language_detect import detect_language

THRESHOLD = float(MODEL["neutral_threshold"])
MAX_LEN = int(MODEL["local_max_length"])
LABEL_ORDER = ["negative", "neutral", "positive"]


def batch_local_predict(texts, batch_size=64):
    tokenizer, model, labels = _load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            idx = logits.argmax(dim=-1)
            for j in range(len(batch)):
                lab = labels[int(idx[j])]
                out.append((lab, float(probs[j][int(idx[j])])))
    return out


def to_3class(label, score):
    label = label.lower().strip()
    if label not in ("positive", "negative"):
        label = "negative"
    return label if score >= THRESHOLD else "neutral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", default=str(SPLITS_DIR / "test.csv"))
    ap.add_argument("--no-translate", action="store_true",
                    help="skip EN->KM translation (raw text goes to the model)")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.split_path, encoding=ENCODING)
    if args.max_rows:
        df = df.head(args.max_rows)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].str.strip().astype(str).tolist()

    model_inputs = []
    detected = []
    translated = 0
    translate_fail = 0

    for t in texts:
        lang = detect_language(t)
        detected.append(lang)
        inp = t
        if lang == "english" and not args.no_translate:
            try:
                inp = translate_en_to_km(t)
                translated += 1
            except Exception:
                translate_fail += 1
        model_inputs.append(inp)

    raw = batch_local_predict(model_inputs)
    y_pred = [to_3class(lab, sc) for lab, sc in raw]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    report = {
        "dataset": str(Path(args.split_path)),
        "rows": len(df),
        "pipeline": "tykea local 2-class + threshold %.2f (production fallback path)" % THRESHOLD,
        "translation": "enabled (best-effort)" if not args.no_translate else "disabled",
        "translated_rows": translated,
        "translate_failures": translate_fail,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            lab: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for lab, p, r, f, s in zip(LABEL_ORDER, *per_class)
        },
        "confusion_matrix": {"rows": LABEL_ORDER, "cols": LABEL_ORDER, "values": cm.tolist()},
    }

    by_lang = {}
    for lang in sorted(set(detected)):
        yt = [g for g, d in zip(y_true, detected) if d == lang]
        yp = [g for g, d in zip(y_pred, detected) if d == lang]
        if yt:
            by_lang[lang] = {
                "rows": len(yt),
                "accuracy": round(float(accuracy_score(yt, yp)), 4),
                "macro_f1": round(float(f1_score(yt, yp, average="macro", labels=LABEL_ORDER)), 4),
            }
    report["by_language"] = by_lang

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "baseline.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nrows={len(df)}  translated={translated}  translate_fail={translate_fail}")
    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}\n")
    print("per-class:")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):")
    print("           " + "  ".join(f"{c:>9}" for c in LABEL_ORDER))
    for r, row in zip(LABEL_ORDER, cm):
        print(f"{r:>9}   " + "  ".join(f"{v:>9}" for v in row))
    print("\nby language:")
    for lang, d in by_lang.items():
        print(f"  {lang:>12}: rows={d['rows']:>4}  acc={d['accuracy']:.4f}  macro_f1={d['macro_f1']:.4f}")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()