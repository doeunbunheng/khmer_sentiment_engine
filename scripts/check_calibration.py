"""Calibration / robustness check for the production 3-class model on val set.

Bins predictions by max-confidence (0.5-0.6, 0.6-0.7, ..., 0.9-1.0), reports
actual accuracy per bin, computes Expected Calibration Error (ECE), and finds
the confidence at which accuracy drops below a target (default 0.80) — the
point where a UI should show "uncertain" instead of a guess.

Usage:
  .venv\\Scripts\\python scripts\\check_calibration.py
  .venv\\Scripts\\python scripts\\check_calibration.py --target-acc 0.80

Output: console summary + reports/calibration.json
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
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.config import SPLITS_DIR, ENCODING, MODEL_DIR

REPORTS_DIR = PROJECT_ROOT / "reports"
MAX_LEN = 256
BATCH_SIZE = 64
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        return {k: torch.tensor(v[i]) for k, v in self.encodings.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", default=str(SPLITS_DIR / "val.csv"))
    ap.add_argument("--model-dir", default=str(MODEL_DIR),
                    help="model dir (default: config.yaml local_model_path, i.e. v2)")
    ap.add_argument("--target-acc", type=float, default=0.80)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--report-name", default="calibration.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    df = pd.read_csv(args.split_path, encoding=ENCODING)
    df = df[df["label"].isin(LABELS)].copy()
    if args.max_rows:
        df = df.head(args.max_rows)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].map(LABEL_TO_ID).tolist()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True)
    loader = DataLoader(TextDataset(enc), batch_size=BATCH_SIZE, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.vstack(probs)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    y_true = np.array(y_true)

    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bin_stats = []
    ece = 0.0
    for lo, hi in bins:
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n == 0:
            bin_stats.append({"bin": f"{lo}-{hi}", "rows": 0, "avg_conf": None,
                              "accuracy": None, "correct": 0})
            continue
        acc = float(accuracy_score(y_true[mask], preds[mask]))
        avg_conf = float(conf[mask].mean())
        ece += (n / len(y_true)) * abs(avg_conf - acc)
        bin_stats.append({"bin": f"{lo}-{hi}", "rows": n,
                          "avg_conf": round(avg_conf, 4), "accuracy": round(acc, 4),
                          "correct": int((preds[mask] == y_true[mask]).sum())})

    # accuracy >= target requires at least this confidence
    thr = None
    if args.target_acc is not None:
        for t in np.arange(0.5, 1.0, 0.01):
            mask = conf >= t
            if mask.sum() > 0 and accuracy_score(y_true[mask], preds[mask]) >= args.target_acc:
                thr = round(float(t), 2)
                break

    report = {
        "model": str(model_dir),
        "dataset": str(Path(args.split_path)),
        "rows": len(df),
        "target_accuracy": args.target_acc,
        "ece": round(ece, 4),
        "overall_accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "bins": bin_stats,
        "min_confidence_for_target_acc": thr,
        "rows_below_target": int((conf < (thr or 1.0)).sum()) if thr else None,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / args.report_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"rows={len(df)}  overall_acc={report['overall_accuracy']:.4f}  "
          f"ECE={report['ece']:.4f}")
    print(f"\n{'bin':>10}  {'rows':>5}  {'avg_conf':>8}  {'accuracy':>8}  correct")
    for b in bin_stats:
        if b["rows"] == 0:
            print(f"{b['bin']:>10}  {'0':>5}  {'—':>8}  {'—':>8}")
            continue
        print(f"{b['bin']:>10}  {b['rows']:>5}  {b['avg_conf']:>8}  "
              f"{b['accuracy']:>8}  {b['correct']}/{b['rows']}")
    if thr is not None:
        print(f"\nconfidence >= {thr:.2f} achieves accuracy >= {args.target_acc}")
        print(f"rows below that confidence (would be flagged uncertain): "
              f"{report['rows_below_target']} ({report['rows_below_target']/len(df)*100:.1f}%)")
    else:
        print(f"\nno confidence threshold reaches accuracy {args.target_acc}")
    print(f"\nsaved: {REPORTS_DIR / args.report_name}")


if __name__ == "__main__":
    main()
