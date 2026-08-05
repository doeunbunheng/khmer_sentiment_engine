import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.prepare_splits import load_labeled, make_splits


def _fixture_df(n=1000):
    labels = ["positive", "neutral", "negative"] * (n // 3)
    labels += ["positive"] * (n - len(labels))
    df = pd.DataFrame(
        {
            "text": [f"អត្ថបទ {i} ល្អ" for i in range(n)],
            "label": labels,
        }
    )
    return df


def test_load_labeled(tmp_path):
    df = _fixture_df(30)
    df.to_csv(tmp_path / "labeled.csv", index=False, encoding="utf-8")
    loaded = load_labeled(tmp_path / "labeled.csv")
    assert len(loaded) == 30
    assert {"text", "label"}.issubset(loaded.columns)


def test_make_splits_ratios():
    df = _fixture_df(3000)
    train, val, test = make_splits(df, seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(train) / total - 0.8) < 0.01
    assert abs(len(val) / total - 0.1) < 0.01
    assert abs(len(test) / total - 0.1) < 0.01


def test_make_splits_stratified():
    df = _fixture_df(3000)
    train, val, test = make_splits(df, seed=42)
    for split in (train, val, test):
        ratios = split["label"].value_counts(normalize=True).to_dict()
        assert abs(ratios["positive"] - 1 / 3) < 0.02
        assert abs(ratios["neutral"] - 1 / 3) < 0.02
        assert abs(ratios["negative"] - 1 / 3) < 0.02


def test_no_overlap():
    df = _fixture_df(3000)
    train, val, test = make_splits(df, seed=42)
    train_ids = set(train["text"])
    assert train_ids.isdisjoint(val["text"])
    assert train_ids.isdisjoint(test["text"])
    assert set(val["text"]).isdisjoint(test["text"])


def test_seed_reproducible():
    df = _fixture_df(3000)
    t1, _, _ = make_splits(df, seed=42)
    t2, _, _ = make_splits(df, seed=42)
    assert t1["text"].tolist() == t2["text"].tolist()
