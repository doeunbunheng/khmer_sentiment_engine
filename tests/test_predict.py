from src.predict import predict_sentiment
from src.preprocessing.anonymize import anonymize_text
from src.common import db

def predict_and_save(text, user_id=None, consent=False):
    result = predict_sentiment(text)
    if consent and user_id is not None and result["language"] != "unknown":
        db.save_feedback(
            user_id=user_id,
            text_orig=result["text"],
            text_anonymized=anonymize_text(result["text"]),
            lang_detect=result["language"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            text_translated=result["translated_text"],
            consent=True,
        )
    return result

def _ok(label, score):
    return label, score


def test_khmer_positive(monkeypatch):
    monkeypatch.setattr("src.predict.classify_khmer", lambda t: _ok("Positive", 0.87))
    out = predict_sentiment("ផលិតផលល្អណាស់")
    assert out["sentiment"] == "positive"
    assert out["confidence"] == 0.87
    assert out["language"] == "khmer"
    assert out["translated_text"] is None


def test_threshold_boundary_neutral(monkeypatch):
    monkeypatch.setattr("src.predict.classify_khmer", lambda t: _ok("Positive", 0.59))
    assert predict_sentiment("ដូចគ្នា")["sentiment"] == "neutral"


def test_threshold_boundary_positive(monkeypatch):
    monkeypatch.setattr("src.predict.classify_khmer", lambda t: _ok("Positive", 0.60))
    assert predict_sentiment("ដូចគ្នា")["sentiment"] == "positive"


def test_english_translated_then_classified(monkeypatch):
    seen = {}

    def fake_translate(text):
        seen["input"] = text
        return "ផលិតផលល្អ"

    monkeypatch.setattr("src.predict.translate_en_to_km", fake_translate)
    monkeypatch.setattr("src.predict.classify_khmer", lambda t: _ok("Positive", 0.8))
    out = predict_sentiment("The product is good")
    assert out["language"] == "english"
    assert seen["input"] == "The product is good"
    assert out["translated_text"] == "ផលិតផលល្អ"
    assert out["sentiment"] == "positive"


def test_mixed_no_translation(monkeypatch):
    calls = {"translate": 0, "classify": 0}

    def fake_translate(text):
        calls["translate"] += 1
        return text

    def fake_classify(text):
        calls["classify"] += 1
        return _ok("Negative", 0.9)

    monkeypatch.setattr("src.predict.translate_en_to_km", fake_translate)
    monkeypatch.setattr("src.predict.classify_khmer", fake_classify)
    out = predict_sentiment("I like it ប៉ុន្តែថ្លៃពេក")
    assert out["language"] == "mixed"
    assert calls["translate"] == 0
    assert out["sentiment"] == "negative"


def test_api_failure_degrades_to_neutral(monkeypatch):
    def boom(text):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.predict.classify_khmer", boom)
    out = predict_sentiment("អត្ថបទណាមួយ")
    assert out["sentiment"] == "neutral"
    assert out["confidence"] == 0.0


def test_empty_input():
    out = predict_sentiment("")
    assert out["sentiment"] == "neutral"
    assert out["language"] == "unknown"

