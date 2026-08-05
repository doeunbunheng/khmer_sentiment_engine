import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean import clean_text


def test_removes_url():
    assert "http://example.com" not in clean_text("ល្អ http://example.com x")
    assert "www.example.com" not in clean_text("www.example.com ល្អ")


def test_removes_email():
    assert clean_text("contact a@b.com now") == "contact now"


def test_removes_mention_and_hashtag():
    assert clean_text("@user ល្អ #sale") == "ល្អ"


def test_collapses_whitespace():
    assert clean_text("ល្អ    ណាស់\n\t") == "ល្អ ណាស់"


def test_keeps_khmer_text():
    text = "ផលិតផលល្អណាស់ តម្លៃសមរម្យ"
    assert clean_text(text) == text


def test_emoji_optional():
    assert clean_text("ល្អ 😀") == "ល្អ 😀"
    assert clean_text("ល្អ 😀", remove_emoji=True) == "ល្អ"


def test_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_mixed_language_survives():
    text = "I like this product ប៉ុន្តែថ្លៃ"
    out = clean_text(text)
    assert "like" in out and "ប៉ុន្តែថ្លៃ" in out
