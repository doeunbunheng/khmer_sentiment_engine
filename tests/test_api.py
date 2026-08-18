"""API tests — FastAPI TestClient, no real server, no model loading, no DB.

- `predict_and_save`, `db.login_user`, `db.fetch_feedback` are mocked.
- Tokens are minted directly via `create_token` (no login round-trip).
- Rate limiter + lockout state are reset before every test.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app, limiter
from src.common.security import create_token, reset_lockout
from src.predict import predict_and_save

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_security_state():
    limiter.reset()
    reset_lockout()


def _auth(user_id=1, role="Admin"):
    return {"Authorization": f"Bearer {create_token(user_id, role)}"}


# ---- health -----------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model"] == "khmer-sentiment-3class-v2"


# ---- auth / login -----------------------------------------------------------

def test_login_success(monkeypatch):
    monkeypatch.setattr(
        "src.common.db.login_user",
        lambda u, p: {"user_id": 1, "role": "Admin", "ok": True},
    )
    r = client.post("/auth/login", json={"username": "demo_admin", "password": "132336BV132336"})
    assert r.status_code == 200
    d = r.json()
    assert d["user_id"] == 1
    assert d["role"] == "Admin"
    assert d["expires_in_seconds"] > 0
    assert d["token"].count(".") == 1


def test_login_wrong_password(monkeypatch):
    monkeypatch.setattr(
        "src.common.db.login_user",
        lambda u, p: {"ok": False},
    )
    r = client.post("/auth/login", json={"username": "demo_admin", "password": "wrong"})
    assert r.status_code == 401


def test_login_lockout_after_five_failures(monkeypatch):
    monkeypatch.setattr(
        "src.common.db.login_user",
        lambda u, p: {"ok": False},
    )
    for _ in range(5):
        r = client.post("/auth/login", json={"username": "victim", "password": "x"})
        assert r.status_code in (401, 429)
    limiter.reset()  # isolate the lockout (not the rate limit)
    r = client.post("/auth/login", json={"username": "victim", "password": "x"})
    assert r.status_code == 429


def test_login_empty_fields_rejected():
    r = client.post("/auth/login", json={"username": "", "password": "x"})
    assert r.status_code == 422
    r = client.post("/auth/login", json={"username": "u", "password": ""})
    assert r.status_code == 422


# ---- auth / register --------------------------------------------------------

def test_register_success(monkeypatch):
    monkeypatch.setattr(
        "src.common.db.register_user",
        lambda **kw: 42,
    )
    r = client.post(
        "/auth/register",
        json={"username": "newbie", "password": "secret123", "full_name": "New User"},
    )
    assert r.status_code == 201
    d = r.json()
    assert d["user_id"] == 42
    assert d["role"] == "User"


def test_register_never_mints_admin(monkeypatch):
    captured = {}

    def fake_register(**kw):
        captured.update(kw)
        return 7

    monkeypatch.setattr("src.common.db.register_user", fake_register)
    r = client.post("/auth/register", json={"username": "newuser", "password": "secret123"})
    assert r.status_code == 201
    assert captured["role"] == "User"


def test_register_conflict_409(monkeypatch):
    def boom(**kw):
        raise Exception("duplicate key")

    monkeypatch.setattr("src.common.db.register_user", boom)
    r = client.post("/auth/register", json={"username": "taken", "password": "secret123"})
    assert r.status_code == 409


def test_register_weak_password_422():
    r = client.post("/auth/register", json={"username": "u", "password": "x"})
    assert r.status_code == 422
    r = client.post("/auth/register", json={"username": "ab", "password": "secret123"})
    assert r.status_code == 422  # username too short (min 3)


# ---- auth guard on predict --------------------------------------------------

def test_predict_requires_token():
    r = client.post("/predict", json={"text": "hello"})
    assert r.status_code == 401


def test_predict_rejects_bad_token():
    r = client.post(
        "/predict",
        json={"text": "hello"},
        headers={"Authorization": "Bearer not.a.token"},
    )
    assert r.status_code == 401


def test_predict_with_valid_token(monkeypatch):
    def fake_predict_and_save(**kw):
        return {
            "text": kw["text"],
            "language": "khmer",
            "sentiment": "positive",
            "confidence": 0.9,
            "translated_text": None,
            "aspects": {
                "business_aspects": {"Price": {"hit": True, "keywords": ["ថោក"]}},
                "emotions": {"scores": {"Joy": 0.8}, "active": ["Joy"]},
            },
        }

    monkeypatch.setattr("src.api.predict_and_save", fake_predict_and_save)
    r = client.post("/predict", json={"text": "ថោក"}, headers=_auth())
    assert r.status_code == 200
    d = r.json()
    assert d["sentiment"] == "positive"
    assert d["aspects"]["business_aspects"]["Price"]["hit"] is True


def test_predict_text_too_long_rejected(monkeypatch):
    r = client.post(
        "/predict",
        json={"text": "x" * 2001},
        headers=_auth(),
    )
    assert r.status_code == 422


def test_predict_with_consent_passes_save_args(monkeypatch):
    captured = {}

    def fake_predict_and_save(**kw):
        captured.update(kw)
        return {
            "text": kw["text"],
            "language": "english",
            "sentiment": "neutral",
            "confidence": 0.5,
            "translated_text": None,
            "aspects": {"business_aspects": {}, "emotions": {"scores": {}, "active": []}},
        }

    monkeypatch.setattr("src.api.predict_and_save", fake_predict_and_save)
    r = client.post(
        "/predict",
        json={"text": "x", "user_id": 7, "consent": True, "names": ["Sok"]},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert captured["user_id"] == 7
    assert captured["consent"] is True
    assert captured["names"] == ["Sok"]


def test_predict_empty_text_ok():
    r = client.post("/predict", json={"text": ""}, headers=_auth())
    assert r.status_code == 200
    assert r.json()["sentiment"] == "neutral"


# ---- feedback admin-only ----------------------------------------------------

def test_feedback_denied_for_user_role():
    r = client.get("/feedback", headers=_auth(role="User"))
    assert r.status_code == 403


def test_feedback_denied_without_token():
    r = client.get("/feedback")
    assert r.status_code == 401


def test_feedback_allowed_for_admin(monkeypatch):
    monkeypatch.setattr(
        "src.common.db.fetch_feedback",
        lambda user_id=None, limit=100: [
            (1, 1, True, "t", "an", "khmer", None, "positive", 0.9, {}, "2026-01-01"),
        ],
    )
    r = client.get("/feedback", headers=_auth(role="Admin"))
    assert r.status_code == 200
    assert r.json()[0]["sentiment"] == "positive"
