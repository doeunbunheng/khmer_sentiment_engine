"""External unseen-data evaluation â€” kh-polarity (ye-kyaw-thu) test on the 3-class model.

The kh-polarity corpus is a human-annotated Khmer polarity corpus (positive /
neutral / negative) with ZERO overlap with the 18,771 training rows â€” a true
generalization check on out-of-domain text (news/political, not food reviews).

Input: data/external_kh_polarity.csv (parsed from kh-polar.ver1.0.txt,
sentence ||| keyword ||| polarity). Output: reports/external_kh_polarity.json.

Usage:
  .venv\\Scripts\\python scripts\\evaluate_external.py
  .venv\\Scripts\\python scripts\\evaluate_external.py --max-rows 200   # smoke
  .venv\\Scripts\\python scripts\\evaluate_external.py --aspects       # + business-aspect stats
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

from src.common.config import ENCODING
from src.models.aspects import detect_business_aspects
from src.preprocessing.language_detect import detect_language

EXTERNAL_CSV = PROJECT_ROOT / "data" / "external_kh_polarity.csv"
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
    ap.add_argument("--dataset", type=Path, default=None,
                    help="CSV with text,label columns (default: full external_kh_polarity.csv)")
    ap.add_argument("--report-name", default="external_kh_polarity.json")
    ap.add_argument("--aspects", action="store_true",
                    help="also compute business-aspect hit statistics")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="model directory (default: models/khmer-sentiment-3class)")
    args = ap.parse_args()

    model_dir = args.model_dir or MODEL_DIR

    print("loading external dataset...")
    df = pd.read_csv(args.dataset or EXTERNAL_CSV, encoding=ENCODING)
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

    report = {
        "model": str(model_dir),
        "dataset": "kh-polarity ver1.0 (ye-kyaw-thu), unseen â€” zero overlap with train/val/test",
        "rows": len(df),
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
    }

    by_lang = {}
    langs = [detect_language(t) for t in texts]
    df = df.assign(_lang=langs)
    for lang, grp in df.groupby("_lang"):
        idx = grp.index.tolist()
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        by_lang[lang] = {
            "rows": len(idx),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
        }
    report["by_language"] = by_lang

    if args.aspects:
        from collections import Counter
        hits = Counter()
        for t in texts:
            for aspect, spec in detect_business_aspects(t).items():
                if spec["hit"]:
                    hits[aspect] += 1
        report["business_aspects"] = {
            aspect: {"rows_hit": n, "pct": round(100.0 * n / len(df), 2)}
            for aspect, n in sorted(hits.items())
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
    if args.aspects:
        print("\nbusiness aspects hit rate:")
        for aspect, d in report["business_aspects"].items():
            print(f"  {aspect:>16}: {d['rows_hit']:>5} rows ({d['pct']}%)")
    print(f"\nsaved: {REPORTS_DIR / args.report_name}")


if __name__ == "__main__":
    main()
