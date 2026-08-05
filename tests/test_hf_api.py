import pytest

from src.models.hf_api import _parse, classify_khmer


def test_parse_valid(monkeypatch):
    payload = [{"label": "Positive", "score": 0.87}]
    assert _parse(payload) == ("Positive", 0.87)


def test_parse_invalid():
    with pytest.raises(ValueError):
        _parse({})


def test_classify_success(monkeypatch):
    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"label": "Negative", "score": 0.91}]

    class FakePost:
        def __init__(self, *a, **k):
            pass

        def post(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr("src.models.hf_api.requests.post", lambda *a, **k: FakeResp())
    assert classify_khmer("មិនល្អទេ", token="t", max_retries=0) == ("Negative", 0.91)


def test_classify_retry_then_success(monkeypatch):
    calls = {"n": 0}

    class FakeResp503:
        status_code = 503

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    class FakeRespOK:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return [{"label": "Positive", "score": 0.99}]

    def fake_post(*a, **k):
        calls["n"] += 1
        return FakeResp503() if calls["n"] < 2 else FakeRespOK()

    monkeypatch.setattr("src.models.hf_api.requests.post", fake_post)
    monkeypatch.setattr("src.models.hf_api.time.sleep", lambda s: None)
    label, score = classify_khmer("ល្អ", token="t", max_retries=2)
    assert (label, score) == ("Positive", 0.99)
    assert calls["n"] == 2


def test_classify_all_fail_falls_back_local(monkeypatch):
    class FakeResp500:
        status_code = 500

        def raise_for_status(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.models.hf_api.requests.post", lambda *a, **k: FakeResp500()
    )
    monkeypatch.setattr("src.models.hf_api.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "src.models.hf_api._call_local", lambda t: ("negative", 0.9)
    )
    assert classify_khmer("ល្អ", token="t", max_retries=1) == ("negative", 0.9)


def test_no_fallback_raises(monkeypatch):
    class FakeResp500:
        status_code = 500

        def raise_for_status(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.models.hf_api.requests.post", lambda *a, **k: FakeResp500()
    )
    monkeypatch.setattr("src.models.hf_api.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        classify_khmer("ល្អ", token="t", max_retries=1, use_local_fallback=False)
