"""Dashboard tests — API client + unseen eval, no network, no Streamlit.

- `ApiClient` HTTP layer is mocked (`requests.request` stubbed).
- Unseen evaluation uses a fake client; the CSV loader uses a temp file.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
import requests

from app.api_client import ApiClient, ApiError
from app.unseen_eval import (
    build_prediction_summary,
    build_report,
    load_unseen_csv,
    normalize_unseen_df,
    parse_input_text,
    run_unseen_eval,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload


def _stub_request(monkeypatch, fake):
    monkeypatch.setattr("app.api_client.requests.request", fake)


# ---- ApiClient: health / auth / predict / feedback -------------------------

def test_client_health(monkeypatch):
    _stub_request(monkeypatch, lambda *a, **k: _FakeResponse(200, {"status": "ok", "db": True}))
    client = ApiClient("http://127.0.0.1:8000")
    assert client.health()["status"] == "ok"


def test_client_login(monkeypatch):
    _stub_request(
        monkeypatch,
        lambda *a, **k: _FakeResponse(200, {"token": "abc.def", "user_id": 1, "role": "Admin"}),
    )
    data = ApiClient("http://x").login("demo_admin", "pw")
    assert data["role"] == "Admin"


def test_client_register_body(monkeypatch):
    captured = {}

    def fake(method, url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(201, {"user_id": 5, "role": "User", "message": "registered"})

    _stub_request(monkeypatch, fake)
    ApiClient("http://x").register("newbie", "secret123", full_name="New User")
    assert captured["json"] == {
        "username": "newbie",
        "password": "secret123",
        "full_name": "New User",
    }
    # optional fields omitted when absent
    captured.clear()
    ApiClient("http://x").register("newbie", "secret123")
    assert "full_name" not in captured["json"]


def test_client_predict_bearer_and_consent(monkeypatch):
    captured = {}

    def fake(method, url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResponse(200, {"sentiment": "positive", "uncertain": False, "confidence": 0.98})

    _stub_request(monkeypatch, fake)
    client = ApiClient("http://x:8000")
    out = client.predict("hi", consent=True, token="tok123")
    assert captured["url"] == "http://x:8000/predict"
    assert captured["headers"]["Authorization"] == "Bearer tok123"
    assert captured["json"] == {"text": "hi", "consent": True}
    assert out["sentiment"] == "positive"


def test_client_predict_omits_bearer_without_token(monkeypatch):
    captured = {}

    def fake(method, url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse(200, {"sentiment": "neutral", "uncertain": True, "confidence": 0.5})

    _stub_request(monkeypatch, fake)
    ApiClient("http://x").predict("hi")
    assert "Authorization" not in captured["headers"]


def test_client_feedback_builds_url(monkeypatch):
    captured = {}

    def fake(method, url, json=None, headers=None, timeout=None):
        captured.update(url=url, headers=headers)
        return _FakeResponse(200, [])

    _stub_request(monkeypatch, fake)
    ApiClient("http://x:8000").feedback(limit=250, token="ttt")
    assert captured["url"] == "http://x:8000/feedback?limit=250"
    assert captured["headers"]["Authorization"] == "Bearer ttt"


def test_network_error_raises_api_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    _stub_request(monkeypatch, boom)
    with pytest.raises(ApiError) as ei:
        ApiClient("http://x").health()
    assert "cannot reach API" in str(ei.value)


def test_http_error_raises_api_error_with_status(monkeypatch):
    _stub_request(
        monkeypatch,
        lambda *a, **k: _FakeResponse(
            409, {"detail": "username, email, or phone already registered"}
        ),
    )
    with pytest.raises(ApiError) as ei:
        ApiClient("http://x").register("u", "secret123")
    assert ei.value.status == 409
    assert "already registered" in str(ei.value)


# ---- unseen_eval ------------------------------------------------------------

def _fake_predict(text):
    sent, rest = text.split(":", 1)
    return {
        "sentiment": sent if sent in ("negative", "neutral", "positive") else "neutral",
        "uncertain": rest.startswith("u"),
        "confidence": 0.55 if rest.startswith("u") else 0.95,
    }


class _FakeClient:
    def __init__(self, seen_tokens=None):
        self.seen_tokens = seen_tokens

    def predict(self, text, consent=False, token=None):
        if self.seen_tokens is not None:
            self.seen_tokens.append(token)
        return _fake_predict(text)


def _result_tuple(text):
    r = _fake_predict(text)
    return (r["sentiment"], r["uncertain"], r["confidence"])


def test_load_unseen_csv_filters_invalid_labels(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("text,label\nhello,negative\nworld,unknown\nx,positive\n", encoding="utf-8")
    df = load_unseen_csv(str(p))
    assert len(df) == 2
    assert set(df["label"]) == {"negative", "positive"}


def test_normalize_unseen_df_aliases(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "Comment, POLARITY \nGreat!,Positive\nMeh,neUtral\nBad,Negative\nok,unknown\n",
        encoding="utf-8",
    )
    df = load_unseen_csv(str(p))
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 3
    assert set(df["label"]) == {"negative", "neutral", "positive"}
    assert df.iloc[0]["text"] == "Great!"


def test_normalize_unseen_df_missing_columns(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        load_unseen_csv(str(p))


def test_normalize_unseen_df_comment_text_without_label(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "Comment_Text\nGreat!\nBad product\nOK\n",
        encoding="utf-8",
    )
    df = load_unseen_csv(str(p))
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 3
    assert df["label"].isna().all()
    assert df.iloc[1]["text"] == "Bad product"


def test_normalize_unseen_df_comment_text_with_label(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "Comment_Text,Sentiment_Label\nGreat!,Positive\nBad product,Negative\n",
        encoding="utf-8",
    )
    df = load_unseen_csv(str(p))
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 2
    assert set(df["label"]) == {"positive", "negative"}


def test_run_unseen_eval_progress_and_results(monkeypatch):
    df = pd.DataFrame(
        {
            "text": ["negative:0", "positive:1", "neutral:u"],
            "label": ["negative", "positive", "neutral"],
        }
    )
    calls = []
    texts, y_true, results = run_unseen_eval(
        _FakeClient(), df, max_rows=None, workers=3,
        progress_cb=lambda d, t: calls.append((d, t)),
    )
    assert y_true == ["negative", "positive", "neutral"]
    assert results[0][0] == "negative"
    assert results[1][2] == 0.95  # confident
    assert results[2][1] is True  # uncertain flag came through
    assert calls and calls[-1] == (3, 3)
    assert len(calls) == 3


def test_run_unseen_eval_caps_rows(monkeypatch):
    df = pd.DataFrame(
        {
            "text": [f"negative:{i}" for i in range(5)],
            "label": ["negative"] * 5,
        }
    )
    texts, y_true, results = run_unseen_eval(_FakeClient(), df, max_rows=2, workers=2)
    assert len(texts) == 2
    assert len(results) == 2


def test_run_unseen_eval_forwards_token(monkeypatch):
    df = pd.DataFrame(
        {
            "text": ["negative:0", "positive:1"],
            "label": ["negative", "positive"],
        }
    )
    seen = []
    run_unseen_eval(_FakeClient(seen_tokens=seen), df, max_rows=None, workers=2, token="tokABC")
    assert seen == ["tokABC", "tokABC"]


def test_build_report_metrics():
    texts = ["negative:0", "positive:u", "neutral:1", "positive:x", "neutral:2"]
    y_true = ["negative", "positive", "positive", "negative", "neutral"]
    results = [_result_tuple(t) for t in texts]

    report = build_report(texts, y_true, results, dataset="d", endpoint="e")
    assert report["rows"] == 5
    assert report["errors"] == 0
    assert report["accuracy"] == 0.6  # rows 3 (true pos) and 4 (true neg) wrong
    assert report["uncertain_analysis"]["uncertain_rows"] == 1
    assert report["uncertain_analysis"]["uncertain_accuracy"] == 1.0
    assert report["uncertain_analysis"]["confident_accuracy"] == 0.5
    assert report["confusion_matrix"]["values"][0][0] == 1
    assert len(report["sample_wrong"]) == 2
    assert len(report["per_class"]) == 3
    assert isinstance(report["by_language"], dict)


def test_build_report_counts_errors():
    texts, y_true = ["neg:0", "neu:1"], ["negative", "neutral"]
    results = [("__error__:boom", False, None), ("neutral", False, 0.9)]
    report = build_report(texts, y_true, results, dataset="d", endpoint="e")
    assert report["errors"] == 1
    assert len(report["sample_errors"]) == 1


def test_parse_input_text():
    block = (
        "negative|bad product\n"
        "positive: really good\n"
        "neutral, just facts\n"
        "no label here\n"
        "Positive|Mixed case\n"
        "\n"
    )
    rows = parse_input_text(block, has_labels=False)
    assert rows == [
        ("negative|bad product", None),
        ("positive: really good", None),
        ("neutral, just facts", None),
        ("no label here", None),
        ("Positive|Mixed case", None),
    ]
    rows = parse_input_text(block, has_labels=True)
    assert rows[0] == ("bad product", "negative")
    assert rows[1] == ("really good", "positive")
    assert rows[2] == ("just facts", "neutral")
    assert rows[3] == ("no label here", None)  # unparseable stays unlabeled
    assert rows[4] == ("Mixed case", "positive")


def test_build_prediction_summary():
    texts = ["A good one", "ទំនិញខូច", "meh"]
    results = [
        ("positive", False, 0.98),
        ("negative", True, 0.55),
        ("neutral", False, 0.92),
        ("__error__:boom", False, None),
    ]
    summary = build_prediction_summary(texts, results)
    assert summary["rows"] == 3
    assert summary["errors"] == 1
    assert summary["uncertain_rows"] == 1
    assert summary["distribution"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert len(summary["predictions"]) == 3
    first = summary["predictions"][0]
    assert first["sentiment"] == "positive"
    assert set(first).issuperset({"row", "text", "language", "confidence", "uncertain"})