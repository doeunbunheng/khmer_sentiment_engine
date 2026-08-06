"""End-to-end unseen-data test through the deployed local API.

Sends every row of data/external_splits/test_ext.csv (989 held-out kh-polarity
rows, never trained on) to the running FastAPI server and compares the API's
sentiment against the true label. Proves the deployed pipeline (server ->
v2 model -> response) matches the offline evaluation (~0.82 accuracy).

The API now requires a bearer token: the script logs in with --user/--password
(default: demo_admin / 132336BV132336 — the seeded demo admin password).

Prerequisites:
  1. Start the server first (raise the predict rate limit for the full run):
     API_PREDICT_LIMIT=1000/minute .venv\\Scripts\\uvicorn src.api:app \\
         --host 127.0.0.1 --port 8000
  2. Then run:
     .venv\\Scripts\\python scripts\\api_test_unseen.py
     .venv\\Scripts\\python scripts\\api_test_unseen.py --max-rows 50   # smoke
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.common.config import ENCODING
from src.preprocessing.language_detect import detect_language

TEST_EXT_CSV = PROJECT_ROOT / "data" / "external_splits" / "test_ext.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
BASE_URL = "http://127.0.0.1:8000"
LABELS = ["negative", "neutral", "positive"]
WORKERS = 8


def fetch(text, headers):
    try:
        r = requests.post(
            f"{BASE_URL}/predict", json={"text": text}, headers=headers, timeout=300
        )
        if r.status_code != 200:
            return f"__error__:{r.status_code}:{r.text[:100]}"
        return r.json().get("sentiment")
    except Exception as e:
        return f"__error__:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows (smoke)")
    ap.add_argument("--report-name", default="api_test_unseen.json")
    ap.add_argument("--user", default="demo_admin")
    ap.add_argument("--password", default="132336BV132336")
    args = ap.parse_args()

    print("checking server...")
    h = requests.get(f"{BASE_URL}/health", timeout=10).json()
    print("health:", h)
    if h.get("status") != "ok":
        raise SystemExit("server not healthy — start it first (uvicorn src.api:app)")

    print("logging in...")
    lr = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": args.user, "password": args.password},
        timeout=10,
    )
    if lr.status_code != 200:
        raise SystemExit(f"login failed ({lr.status_code}): {lr.text}")
    token = lr.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"logged in as {args.user} (role={lr.json()['role']})")

    df = pd.read_csv(TEST_EXT_CSV, encoding=ENCODING)
    df = df[df["label"].isin(LABELS)].copy()
    if args.max_rows:
        df = df.head(args.max_rows)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].map({lab: i for i, lab in enumerate(LABELS)}).tolist()

    print(f"sending {len(texts)} rows to {BASE_URL} ...")
    preds = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch, t, headers): i for i, t in enumerate(texts)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            preds[i] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(texts)}")

    errors = [p for p in preds if isinstance(p, str) and p.startswith("__error__")]
    if errors:
        print(f"ERRORS: {len(errors)} rows failed, e.g.: {errors[0][:80]}")
    y_pred = [
        LABELS.index(p) if p in LABELS else 2
        for p in preds
    ]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    by_lang = {}
    for lang in ("khmer", "english", "mixed"):
        idx = [i for i, t in enumerate(texts) if detect_language(t) == lang]
        if not idx:
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        by_lang[lang] = {
            "rows": len(idx),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
        }

    report = {
        "endpoint": f"{BASE_URL}/predict",
        "model": "khmer-sentiment-3class-v2 (served by API)",
        "dataset": "data/external_splits/test_ext.csv (held-out, never trained on)",
        "rows": len(df),
        "errors": len(errors),
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

    print(f"\nrows={len(df)} errors={len(errors)}")
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
