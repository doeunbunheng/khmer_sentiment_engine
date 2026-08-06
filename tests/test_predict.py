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
    monkeypatch.setattr("src.predict.local_predict", lambda t: _ok("positive", 0.87))
    out = predict_sentiment("ផលិតផលល្អណាស់")
    assert out["sentiment"] == "positive"
    assert out["confidence"] == 0.87
    assert out["language"] == "khmer"
    assert out["translated_text"] is None


def test_uncertain_flag_high_confidence(monkeypatch):
    monkeypatch.setattr("src.predict.local_predict", lambda t: _ok("positive", 0.97))
    out = predict_sentiment("ផលិតផលល្អណាស់")
    assert out["confidence"] == 0.97
    assert out["uncertain"] is False


def test_uncertain_flag_low_confidence(monkeypatch):
    monkeypatch.setattr("src.predict.local_predict", lambda t: _ok("positive", 0.55))
    out = predict_sentiment("ប្រហែលល្អ")
    assert out["confidence"] == 0.55
    assert out["uncertain"] is True


def test_uncertain_flag_boundary(monkeypatch):
    from src.common.config import UNCERTAINTY_THRESHOLD

    monkeypatch.setattr(
        "src.predict.local_predict", lambda t: _ok("positive", UNCERTAINTY_THRESHOLD)
    )
    out = predict_sentiment("ដូចគ្នា")
    assert out["uncertain"] is False


def test_uncertain_flag_failure_path(monkeypatch):
    monkeypatch.setattr(
        "src.predict.local_predict",
        lambda t: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = predict_sentiment("អ្វីមួយ")
    assert out["confidence"] == 0.0
    assert out["uncertain"] is True


def test_uncertain_flag_empty_text():
    out = predict_sentiment("   ")
    assert out["uncertain"] is True


def test_model_neutral_direct(monkeypatch):
    # neutral now comes from the 3-class model itself, not a threshold rule
    monkeypatch.setattr("src.predict.local_predict", lambda t: _ok("neutral", 0.72))
    out = predict_sentiment("ដូចគ្នា")
    assert out["sentiment"] == "neutral"
    assert out["confidence"] == 0.72


def test_model_low_confidence_keeps_label(monkeypatch):
    # no threshold: low confidence still returns the model's label
    monkeypatch.setattr("src.predict.local_predict", lambda t: _ok("positive", 0.41))
    out = predict_sentiment("ប្រហែលល្អ")
    assert out["sentiment"] == "positive"
    assert out["confidence"] == 0.41


def test_english_classified_directly(monkeypatch):
    # the 3-class model handles English natively — no translation
    calls = {"predict": 0}

    def fake_predict(text):
        calls["predict"] += 1
        assert text == "The product is good"
        return _ok("positive", 0.8)

    monkeypatch.setattr("src.predict.local_predict", fake_predict)
    out = predict_sentiment("The product is good")
    assert out["language"] == "english"
    assert out["sentiment"] == "positive"
    assert calls["predict"] == 1


def test_mixed_classified_directly(monkeypatch):
    monkeypatch.setattr("src.predict.local_predict", lambda t: _ok("negative", 0.9))
    out = predict_sentiment("I like it ប៉ុន្តែថ្លៃពេក")
    assert out["language"] == "mixed"
    assert out["sentiment"] == "negative"


def test_model_failure_degrades_to_neutral(monkeypatch):
    def boom(text):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.predict.local_predict", boom)
    out = predict_sentiment("អត្ថបទណាមួយ")
    assert out["sentiment"] == "neutral"
    assert out["confidence"] == 0.0


def test_unknown_label_degrades_to_neutral(monkeypatch):
    monkeypatch.setattr("src.predict.local_predict", lambda t: _ok("weird", 0.9))
    assert predict_sentiment("សាកល្បង")["sentiment"] == "neutral"


def test_empty_input():
    out = predict_sentiment("")
    assert out["sentiment"] == "neutral"
    assert out["language"] == "unknown"
