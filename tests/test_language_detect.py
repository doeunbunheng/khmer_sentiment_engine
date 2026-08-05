from src.preprocessing.language_detect import detect_language


def test_khmer_only():
    assert detect_language("ផលិតផលល្អណាស់ តម្លៃសមរម្យ") == "khmer"


def test_english_only():
    assert detect_language("I like this product") == "english"


def test_mixed():
    assert detect_language("I like this product ប៉ុន្តែថ្លៃពេក") == "mixed"


def test_numbers_and_symbols():
    assert detect_language("12345 !!!") == "unknown"


def test_empty():
    assert detect_language("") == "unknown"
    assert detect_language(None) == "unknown"
