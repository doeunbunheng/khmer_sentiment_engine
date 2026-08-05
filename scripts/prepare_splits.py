"""Split labeled data into train/val/test (80/10/10, stratified by label)."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import (
    ENCODING,
    LABELED_DIR,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.preprocessing.clean import clean_text
from src.preprocessing.segment import segment_text


def load_labeled(path):
    df = pd.read_csv(path, encoding=ENCODING)
    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    df = df.rename(columns={text_col: "text"})
    if "segmented_text" not in df.columns:
        df["segmented_text"] = df["text"].map(segment_text)
    df = df[df["text"].notna() & df["text"].str.strip().ne("")]
    df = df.drop_duplicates(subset="text")
    df = df[df["label"].notna()]
    df = df.reset_index(drop=True)
    return df


def make_splits(df, seed):
    remaining = TRAIN_RATIO + VAL_RATIO
    train, temp = train_test_split(
        df, test_size=1 - TRAIN_RATIO, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        temp,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        stratify=temp["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, default=None, help="Labeled CSV path")
    parser.add_argument("--out", type=Path, default=SPLITS_DIR, help="Output dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = [args.labeled] if args.labeled else sorted(LABELED_DIR.glob("*.csv"))
    if not sources or sources[0] is None:
        sys.exit(f"No labeled CSV found in {LABELED_DIR}. Drop a CSV into data/labeled/ or pass --labeled.")

    frames = []
    for src in sources:
        if not src.exists():
            sys.exit(f"Labeled file not found: {src}")
        frames.append(load_labeled(src))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    train, val, test = make_splits(df, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out / "train.csv", index=False, encoding=ENCODING)
    val.to_csv(args.out / "val.csv", index=False, encoding=ENCODING)
    test.to_csv(args.out / "test.csv", index=False, encoding=ENCODING)

    report = {
        "source_files": [str(s) for s in sources],
        "total_after_clean": len(df),
        "splits": {
            "train": {"rows": len(train), "labels": train["label"].value_counts().to_dict()},
            "val": {"rows": len(val), "labels": val["label"].value_counts().to_dict()},
            "test": {"rows": len(test), "labels": test["label"].value_counts().to_dict()},
        },
    }
    with open(args.out / "splits_report.json", "w", encoding=ENCODING) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
