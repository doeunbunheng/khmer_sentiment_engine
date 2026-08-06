"""Tests for src/models/aspects.py — rules never touch the network; the
songhieng model path is tested with a fake `_load` (see tests/conftest.py).
"""

import pytest

from src.models.aspects import (
    THRESHOLD,
    detect_business_aspects,
    predict,
    predict_emotions,
)
from src.predict import predict_sentiment

ASPECTS = ("Price", "Service", "Product Quality", "Authenticity", "Delivery")


def _all_empty(business):
    assert set(business) == set(ASPECTS)
    for spec in business.values():
        assert spec == {"hit": False, "keywords": []}


def _hit(business, aspect):
    return business[aspect]["hit"], business[aspect]["keywords"]


# ---- business aspect rules (no model involved) -----------------------------

@pytest.mark.parametrize("text,aspect,keyword", [
    ("ថ្លៃពេកណាស់", "Price", "ថ្លៃ"),
    ("the price is too high", "Price", "price"),
    ("ថោកជាងគេ", "Price", "ថោក"),
    ("bad discount", "Price", "discount"),
    ("បង់លុយច្រើនពេក", "Price", "បង់"),
    ("ការប្រូម៉ូសិនល្អ", "Price", "ប្រូម៉ូសិន"),
    ("what a great deal", "Price", "deal"),
    ("I pay too much", "Price", "pay"),
])
def test_price_rules(text, aspect, keyword):
    hit, kws = _hit(detect_business_aspects(text), aspect)
    assert hit
    assert keyword in kws


@pytest.mark.parametrize("text,aspect,keyword", [
    ("សេវាកម្មល្អ", "Service", "សេវា"),
    ("friendly staff", "Service", "staff"),
    ("customer support is slow", "Service", "support"),
    ("អតិថិជនពេញចិត្ត", "Service", "អតិថិជន"),
    ("អ្នកលក់ចិត្តល្អ", "Service", "អ្នកលក់"),
    ("ម្ចាស់ហាងមានចិត្តល្អ", "Service", "ម្ចាស់ហាង"),
    ("the seller was nice", "Service", "seller"),
    ("helpful owner", "Service", "owner"),
])
def test_service_rules(text, aspect, keyword):
    hit, kws = _hit(detect_business_aspects(text), aspect)
    assert hit
    assert keyword in kws


@pytest.mark.parametrize("text,aspect,keyword", [
    ("គុណភាពល្អណាស់", "Product Quality", "គុណភាព"),
    ("ផលិតផលប្រើបានយូរ", "Product Quality", "ផលិតផល"),
    ("durable product", "Product Quality", "durable"),
    ("ប្រើប្រាស់បានល្អ", "Product Quality", "ប្រើប្រាស់"),
    ("ទំនិញខូច", "Product Quality", "ខូច"),
    ("ទំនិញនាំចូល", "Product Quality", "នាំចូល"),
    ("រឹងមាំប្រើបានយូរ", "Product Quality", "រឹងមាំ"),
    ("the item was broken", "Product Quality", "broken"),
])
def test_product_quality_rules(text, aspect, keyword):
    hit, kws = _hit(detect_business_aspects(text), aspect)
    assert hit
    assert keyword in kws


@pytest.mark.parametrize("text,aspect,keyword", [
    ("របស់ក្លែងក្លាយ", "Authenticity", "ក្លែងក្លាយ"),
    ("is this original", "Authenticity", "original"),
    ("counterfeit item", "Authenticity", "counterfeit"),
    ("របស់ចម្លង", "Authenticity", "ចម្លង"),
    ("របស់ត្រាប់", "Authenticity", "ត្រាប់"),
    ("របស់ហ្វូឡឹប", "Authenticity", "ហ្វូឡឹប"),
    ("a cheap knockoff", "Authenticity", "knockoff"),
    ("this is a dupe", "Authenticity", "dupe"),
])
def test_authenticity_rules(text, aspect, keyword):
    hit, kws = _hit(detect_business_aspects(text), aspect)
    assert hit
    assert keyword in kws


@pytest.mark.parametrize("text,aspect,keyword", [
    ("ដឹកជញ្ជូនយឺត", "Delivery", "ដឹកជញ្ជូន"),
    ("delivery was late", "Delivery", "delivery"),
    ("where is my parcel", "Delivery", "parcel"),
    ("បញ្ជាទំនិញរួច", "Delivery", "បញ្ជា"),
    ("ទំនិញមកដល់ហើយ", "Delivery", "មកដល់"),
    ("ដឹកមកដល់ផ្ទះ", "Delivery", "ដល់ផ្ទះ"),
    ("my order arrived", "Delivery", "order"),
    ("track my shipment", "Delivery", "shipment"),
])
def test_delivery_rules(text, aspect, keyword):
    hit, kws = _hit(detect_business_aspects(text), aspect)
    assert hit
    assert keyword in kws


def test_multi_label_comment_hits_multiple():
    out = detect_business_aspects("delivery is slow and the price is too high")
    assert out["Delivery"]["hit"]
    assert out["Price"]["hit"]
    assert not out["Service"]["hit"]


def test_no_keyword_text_all_empty():
    _all_empty(detect_business_aspects("just a random greeting"))


def test_empty_text_all_empty():
    _all_empty(detect_business_aspects(""))
    _all_empty(detect_business_aspects(None))


# ---- emotions (fake model) --------------------------------------------------

def _fake_load():
    class FakeModel:
        config = type("Cfg", (), {"num_labels": 8})()

        def __call__(self, **kw):
            import torch
            logits = torch.tensor([
                [-2.0, -1.5, -1.0, -0.5, 1.5, -1.5, -1.0, -0.5]
            ])
            return type("Out", (), {"logits": logits})()

    return (
        type("Tk", (), {"__call__": lambda self, *a, **k: {"input_ids": None}})(),
        FakeModel(),
        ("Anger", "Anticipation", "Disgust", "Fear", "Joy", "Optimism", "Sadness", "Surprise"),
    )


def test_emotions_scores_and_active(monkeypatch):
    monkeypatch.setattr("src.models.aspects._load", _fake_load)
    out = predict_emotions("រឿងល្អ")
    assert set(out["scores"]) == {
        "Anger", "Anticipation", "Disgust", "Fear", "Joy", "Optimism", "Sadness", "Surprise",
    }
    assert all(0.0 <= v <= 1.0 for v in out["scores"].values())
    assert "Joy" in out["active"]  # sigmoid(1.5) >= 0.5
    assert "Anger" not in out["active"]  # sigmoid(0.1) < 0.5


def test_emotions_threshold_constant_sane():
    assert 0.0 < THRESHOLD < 1.0


def test_emotions_empty_text_no_load():
    # _load is patched to raise by conftest — must not be called for empty text
    assert predict_emotions("") == {"scores": {}, "active": []}


def test_emotions_load_failure_raises_strict():
    # conftest makes _load raise — predict_emotions is strict; predict() degrades
    with pytest.raises(RuntimeError):
        predict_emotions("some text here")


def test_combined_predict_degrades_on_load_failure():
    # conftest makes _load raise -> predict() must return empty emotions
    out = predict("some text here")
    assert out["emotions"] == {"scores": {}, "active": []}


def test_emotions_label_count_mismatch_raises(monkeypatch):
    import torch

    class FakeModel:
        config = type("Cfg", (), {"num_labels": 3})()

        def __call__(self, **kw):
            return type("Out", (), {"logits": torch.tensor([[0.0] * 8])})()

        def eval(self):
            return self

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        lambda *a, **k: FakeModel(),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: type(
            "Tk", (), {"__call__": lambda self, *a, **k: {}}
        )(),
    )
    monkeypatch.setattr(
        "src.models.aspects.ASPECT_MODEL_DIR",
        type("Dir", (), {"exists": lambda self: True})(),
    )
    with pytest.raises(ValueError):
        predict_emotions("anything")


# ---- combined predict() -----------------------------------------------------

def test_predict_combines_both(monkeypatch):
    monkeypatch.setattr("src.models.aspects._load", _fake_load)
    out = predict("delivery was slow, price too high")
    assert out["business_aspects"]["Delivery"]["hit"]
    assert out["business_aspects"]["Price"]["hit"]
    assert set(out["emotions"]["scores"]) == {
        "Anger", "Anticipation", "Disgust", "Fear", "Joy", "Optimism", "Sadness", "Surprise",
    }


def test_predict_failure_safe(monkeypatch):
    monkeypatch.setattr("src.models.aspects.detect_business_aspects", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    out = predict("whatever")
    assert out["business_aspects"] == {
        a: {"hit": False, "keywords": []} for a in ASPECTS
    }
    assert out["emotions"] == {"scores": {}, "active": []}


# ---- integration with predict_sentiment -------------------------------------

def test_predict_sentiment_includes_aspects(monkeypatch):
    monkeypatch.setattr("src.predict.local_predict", lambda t: ("positive", 0.9))
    monkeypatch.setattr("src.predict.predict_aspects", lambda t: {
        "business_aspects": detect_business_aspects(t),
        "emotions": {"scores": {}, "active": []},
    })
    out = predict_sentiment("ផលិតផលល្អ តម្លៃថោក")
    assert out["aspects"]["business_aspects"]["Price"]["hit"]
    assert out["aspects"]["business_aspects"]["Product Quality"]["hit"]


def test_predict_sentiment_empty_has_empty_aspects():
    out = predict_sentiment("")
    assert out["aspects"] == {
        "business_aspects": {}, "emotions": {"scores": {}, "active": []}
    }


def test_predict_sentiment_aspect_failure_still_predicts(monkeypatch):
    monkeypatch.setattr("src.predict.local_predict", lambda t: ("negative", 0.85))
    monkeypatch.setattr("src.predict.predict_aspects", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    out = predict_sentiment("delivery too slow")
    assert out["sentiment"] == "negative"
    assert out["confidence"] == 0.85
