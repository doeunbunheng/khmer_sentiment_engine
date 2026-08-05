"""Tests for phase-script pure logic — no GPU, no model loading, no network.

Covers:
  - evaluate_baseline.to_3class()     threshold mapping (tykea 2-class -> 3-class)
  - stack_phase3.tykea_to_3probs()    (label, score) -> [pos, neu, neg] math
  - reports/*.json schema validation (required keys, numbers in range)
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
REPORTS = PROJECT_ROOT / "reports"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_baseline import THRESHOLD, to_3class  # noqa: E402
from stack_phase3 import tykea_to_3probs  # noqa: E402


# ---- to_3class threshold mapping ------------------------------------------

def test_to3_positive_above_threshold():
    assert to_3class("Positive", THRESHOLD + 0.05) == "positive"


def test_to3_negative_above_threshold():
    assert to_3class("Negative", THRESHOLD + 0.05) == "negative"


def test_to3_below_threshold_is_neutral():
    assert to_3class("Positive", THRESHOLD - 0.05) == "neutral"
    assert to_3class("Negative", THRESHOLD - 0.05) == "neutral"


def test_to3_exactly_at_threshold_is_classified():
    assert to_3class("Positive", THRESHOLD) == "positive"


def test_to3_unknown_label_defaults_negative():
    assert to_3class("LABEL_0", 0.9) == "negative"


def test_to3_case_insensitive():
    assert to_3class("pOsItIvE", 0.9) == "positive"
    assert to_3class("  positive  ", 0.9) == "positive"


# ---- tykea_to_3probs pseudo-probability math -------------------------------

def test_probs_sum_to_one():
    for label, score in [("Positive", 0.9), ("Negative", 0.7),
                         ("positive", 0.5), ("negative", 0.99)]:
        p = tykea_to_3probs(label, score)
        assert len(p) == 3
        assert abs(sum(p) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in p)


def test_positive_high_score():
    p = tykea_to_3probs("Positive", 0.95)
    assert p == pytest.approx([0.95 / 1.05, 0.05 / 1.05, 0.05 / 1.05])


def test_negative_high_score():
    p = tykea_to_3probs("Negative", 0.9)
    assert p == pytest.approx([0.1 / 1.1, 0.1 / 1.1, 0.9 / 1.1])


def test_uncertain_score_gives_high_neutral():
    p = tykea_to_3probs("Positive", 0.51)
    assert p[1] == pytest.approx(0.49 / 1.49)  # neutral grows as confidence falls
    # neutral mass grows monotonically as confidence falls toward 0.5
    low = tykea_to_3probs("Positive", 0.51)
    high = tykea_to_3probs("Positive", 0.95)
    assert low[1] > high[1]


def test_label_order_pos_neu_neg():
    p = tykea_to_3probs("Negative", 0.8)
    assert p[0] < p[2]  # pos < neg
    p2 = tykea_to_3probs("Positive", 0.8)
    assert p2[0] > p2[2]  # pos > neg


# ---- report JSON schema validation -----------------------------------------

REQUIRED_KEYS = {
    "baseline.json": {"accuracy", "macro_f1", "per_class", "confusion_matrix"},
    "phase2_val.json": {"val", "per_class", "confusion_matrix", "by_language"},
    "phase3_val.json": {"val", "per_class", "confusion_matrix", "by_language"},
    "phase4_test.json": {"accuracy", "macro_f1", "per_class", "confusion_matrix", "by_language"},
    "english_direct_vs_translate.json": {"results", "rows", "conclusion"},
}


@pytest.mark.parametrize("name,keys", REQUIRED_KEYS.items())
def test_report_has_required_keys(name, keys):
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = keys - set(data.keys())
    assert not missing, f"{name} missing keys: {missing}"


def test_all_report_metrics_in_range():
    for name in REQUIRED_KEYS:
        path = REPORTS / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        accs = []
        if "accuracy" in data:
            accs.append(data["accuracy"])
        if "macro_f1" in data:
            accs.append(data["macro_f1"])
        if "val" in data and isinstance(data["val"], dict):
            for k in ("eval_accuracy", "eval_macro_f1"):
                if k in data["val"]:
                    accs.append(data["val"][k])
        for v in accs:
            assert 0.0 <= v <= 1.0, f"{name} metric out of range: {v}"
        for label, pc in data.get("per_class", {}).items():
            for k in ("precision", "recall", "f1"):
                v = pc[k]
                assert 0.0 <= v <= 1.0, f"{name} {label}.{k} out of range: {v}"
