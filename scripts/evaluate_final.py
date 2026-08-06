"""Phase 4 — One-shot final evaluation of the production 3-class model on test.csv.

Model: models/khmer-sentiment-3class (xlm-roberta-base fine-tuned, Phase 2).
Raw text in (no translation — the 3-class model handles Khmer/English/mixed natively).
Outputs: accuracy, macro-F1, per-class table, confusion matrix, per-language
breakdown -> reports/phase4_test.json. This is the FINAL published number.

Usage:
  .venv\\Scripts\\python scripts\\evaluate_final.py
  .venv\\Scripts\\python scripts\\evaluate_final.py --max-rows 200   # smoke
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
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.config import SPLITS_DIR, ENCODING, SEED

MODEL_DIR = PROJECT_ROOT / "models" / "khmer-sentiment-3class"
REPORTS_DIR = PROJECT_ROOT / "reports"
MAX_LEN = 256
BATCH_SIZE = 64
LABELS = ["negative", "neutral", "positive"]


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.encodings.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows (smoke test)")
    ap.add_argument("--report-name", default="phase4_test.json")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="model directory (default: models/khmer-sentiment-3class)")
    args = ap.parse_args()

    model_dir = args.model_dir or MODEL_DIR

    print("loading test split...")
    df = pd.read_csv(SPLITS_DIR / "test.csv", encoding=ENCODING)
    df = df[df["label"].isin(LABELS)].copy()
    if args.max_rows:
        df = df.head(args.max_rows)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].map({lab: i for i, lab in enumerate(LABELS)}).tolist()

    print(f"loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True)
    loader = DataLoader(TextDataset(enc), batch_size=BATCH_SIZE, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.vstack(probs)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    by_lang = {}
    for lang, mask in df.groupby("language").groups.items():
        idx = sorted(mask)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        by_lang[lang] = {
            "rows": len(idx),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
        }

    report = {
        "model": str(model_dir),
        "dataset": str(SPLITS_DIR / "test.csv"),
        "rows": len(df),
        "seed": SEED,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            lab: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for lab, p, r, f, s in zip(LABELS, *per_class)
        },
        "confusion_matrix": {
            "rows": LABELS, "cols": LABELS, "values": cm.tolist()
        },
        "by_language": by_lang,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / args.report_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nrows={len(df)}")
    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}\n")
    print("per-class:")
    for lab, d in report["per_class"].items():
        print(f"  {lab:>8}: prec={d['precision']:.4f}  rec={d['recall']:.4f}  f1={d['f1']:.4f}  n={d['support']}")
    print("\nconfusion matrix (rows=true, cols=pred):")
    print("           " + "  ".join(f"{c:>9}" for c in LABELS))
    for r, row in zip(LABELS, cm):
        print(f"{r:>9}   " + "  ".join(f"{v:>9}" for v in row))
    print("\nby language:")
    for lang, d in by_lang.items():
        print(f"  {lang:>12}: rows={d['rows']:>4}  acc={d['accuracy']:.4f}  macro_f1={d['macro_f1']:.4f}")
    print(f"\nsaved: {REPORTS_DIR / args.report_name}")


if __name__ == "__main__":
    main()
