"""AI agent tests - offline explainer, no network, no Streamlit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.ai_agent import chat, explain

PRED = {
    "text": "The delivery was late and the price was too high",
    "language": "english",
    "sentiment": "negative",
    "confidence": 0.71,
    "uncertain": True,
    "aspects": {
        "business_aspects": {
            "Delivery": {"hit": True, "keywords": ["late", "delivery"]},
            "Price": {"hit": True, "keywords": ["price"]},
        },
        "emotions": {"scores": {"Anger": 0.8}, "active": ["Anger"]},
    },
}

DATASET = {
    "type": "dataset",
    "rows": 989,
    "errors": 0,
    "accuracy": 0.8241,
    "macro_f1": 0.7543,
    "per_class": {
        "negative": {"f1": 0.829},
        "neutral": {"f1": 0.570},
        "positive": {"f1": 0.864},
    },
    "uncertain_analysis": {"uncertain_rows": 0},
    "distribution": {"negative": 200, "neutral": 100, "positive": 689},
}


def test_no_context_explainable():
    out = explain("why", None)
    assert "don't have a result" in out


def test_why_negative():
    out = explain("why is this negative?", PRED)
    assert "negative" in out
    assert "Delivery" in out and "Price" in out
    assert "71%" in out or "0.71" in out


def test_uncertain_question():
    out = explain("is it uncertain? should I trust it?", PRED)
    assert "uncertain" in out
    assert "needs review" in out


def test_aspects_question():
    out = explain("what topics is this about?", PRED)
    assert "Delivery" in out and "Price" in out


def test_emotions_question():
    out = explain("what emotions fired?", PRED)
    assert "Anger" in out


def test_metrics_question():
    out = explain("what are the metrics?", DATASET)
    assert "82.4" in out
    assert "macro-f1" in out.lower()


def test_what_happened_single_prediction():
    out = explain("what happened on this result?", PRED)
    assert "negative" in out
    assert "Delivery" in out and "Price" in out
    assert "71%" in out or "0.71" in out


def test_why_did_it_show_single_prediction():
    out = explain("why did it show like this?", PRED)
    assert "negative" in out
    assert "Delivery" in out
    assert "no dataset-wide metrics" not in out


def test_metrics_question_single_prediction_explains_result():
    out = explain("what are the metrics?", PRED)
    assert "negative" in out
    assert "no dataset-wide metrics" not in out


def test_clear_summary_dataset():
    out = explain("give me a short summary", DATASET)
    assert "accuracy" in out.lower()


def test_chat_falls_back_to_local_without_api(monkeypatch):
    monkeypatch.delenv("AGENT_API_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    out = chat("why positive?", PRED)
    assert isinstance(out, str)
    assert out


def test_chat_uses_llm_when_configured(monkeypatch):
    monkeypatch.setenv("AGENT_API_URL", "http://fake/v1")
    monkeypatch.setenv("AGENT_API_KEY", "k")

    def fake_post(url, headers=None, json=None, timeout=None):
        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "LLM ANSWER"}}]}

        return R()

    monkeypatch.setattr("app.ai_agent.requests.post", fake_post)
    assert chat("why?", PRED) == "LLM ANSWER"


def test_dataset_generic_question_answers_benchmark():
    out = explain("what's happen on this dataset", DATASET)
    assert "82.4" in out
    assert "Per class" in out


def test_dataset_without_accuracy_explains_prediction_overview():
    unlabeled = {
        "type": "dataset",
        "rows": 31,
        "accuracy": None,
        "macro_f1": None,
        "distribution": {"negative": 11, "neutral": 9, "positive": 11},
        "uncertain_rows": 0,
    }
    out = explain("what's happen on this dataset", unlabeled)
    assert "no true labels" in out
    assert "Predicted labels" in out


def test_dataset_improve_question_advises():
    out = explain("what should I do next?", DATASET)
    assert "uncertain" in out.lower()


def test_unknown_question_gets_full_digest_prediction():
    out = explain("what else can you tell me", PRED)
    assert "negative" in out
    assert "language" in out.lower()
    assert "Delivery" in out


def test_unknown_question_gets_full_digest_dataset():
    out = explain("random chatty line", DATASET)
    assert "82.4" in out
    assert "To improve this" in out


def test_chat_override_wins_over_env_when_failing(monkeypatch):
    monkeypatch.delenv("AGENT_API_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    out = chat("why positive?", PRED, url=None, key=None, model=None)
    assert isinstance(out, str)
    assert "positive" in out


def test_chat_llm_overrides_from_ui(monkeypatch):
    monkeypatch.delenv("AGENT_API_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)

    def fake_post(url, headers=None, json=None, timeout=None):
        assert json["model"] == "gemini-2.0-flash"
        assert "Bearer ui-key" in headers["Authorization"]

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "UI LLM"}}]}

        return R()

    monkeypatch.setattr("app.ai_agent.requests.post", fake_post)
    assert chat(
        "why?", PRED,
        url="https://example/v1", key="ui-key", model="gemini-2.0-flash",
    ) == "UI LLM"