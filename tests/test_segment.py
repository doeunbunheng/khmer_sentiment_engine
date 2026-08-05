import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.preprocessing.segment as segment_mod
from src.preprocessing.segment import segment_text


def test_empty():
    assert segment_text("") == ""
    assert segment_text(None) == ""


def test_khmer_returns_tokens():
    out = segment_text("ផលិតផលល្អណាស់")
    assert isinstance(out, str)
    assert len(out.split()) > 0


def test_fallback_on_missing_khmernltk(monkeypatch):
    monkeypatch.setattr(segment_mod, "_HAVE_KHMERNLTK", False)
    assert segment_text("ផលិតផល ល្អណាស់") == "ផលិតផល ល្អណាស់"


def test_english_unaffected():
    out = segment_text("I like this product")
    assert out == "I like this product"
