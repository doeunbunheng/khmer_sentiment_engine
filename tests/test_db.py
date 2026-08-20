import pytest

from src.common import db
from src.preprocessing.anonymize import anonymize_text

try:
    db.connect().close()
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DB_AVAILABLE, reason="PostgreSQL not reachable — start the DB first"
)


@pytest.fixture
def user_id():
    uid = db.login_user("demo_admin", "Secret123!")["user_id"]
    return uid


def test_save_feedback_returns_id(user_id):
    fid = db.save_feedback(
        user_id,
        text_orig="Call me 012-345-678",
        lang_detect="english",
        sentiment="positive",
        confidence=0.8,
        consent=True,
    )
    assert fid is not None


def test_saved_row_has_anonymized_text(user_id):
    fid = db.save_feedback(
        user_id=user_id,
        text_orig="email a@b.com please",
        lang_detect="english",
        sentiment="neutral",
        confidence=0.4,
        consent=True,
    )
    rows = db.fetch_feedback(user_id=user_id, limit=500)
    row = next(r for r in rows if r[0] == fid)
    stored = dict(
        zip(
            [
                "id",
                "user_id",
                "consent_granted",
                "text_orig",
                "text_anonymized",
                "lang_detect",
                "text_translated",
                "sentiment",
                "confidence",
                "aspects",
                "created_at",
            ],
            row,
        )
    )
    assert stored["consent_granted"] is True
    assert "a@b.com" not in stored["text_anonymized"]
    assert stored["text_anonymized"] == anonymize_text("email a@b.com please")


def test_consent_false_raises(user_id):
    with pytest.raises(ValueError):
        db.save_feedback(
            user_id=user_id,
            text_orig="no consent here",
            lang_detect="english",
            sentiment="positive",
            confidence=0.7,
            consent=False,
        )


def test_saved_row_roundtrips_aspects_json(user_id):
    aspects = {
        "business_aspects": {
            "Price": {"hit": True, "keywords": ["price"]},
            "Service": {"hit": False, "keywords": []},
        },
        "emotions": {"scores": {"Joy": 0.8, "Anger": 0.1}, "active": ["Joy"]},
    }
    fid = db.save_feedback(
        user_id=user_id,
        text_orig="price is great",
        lang_detect="english",
        sentiment="positive",
        confidence=0.9,
        aspects=aspects,
        consent=True,
    )
    rows = db.fetch_feedback(user_id=user_id, limit=500)
    row = next(r for r in rows if r[0] == fid)
    assert row[9] == aspects


def test_log_analysis_returns_id(user_id):
    aid = db.log_analysis(
        user_id,
        text="Call me 012-345-678",
        language="english",
        sentiment="positive",
        confidence=0.81,
    )
    assert aid is not None


def test_logged_analysis_has_anonymized_text(user_id):
    aid = db.log_analysis(
        user_id,
        text="email a@b.com please  https://example.com/x",
        language="english",
        sentiment="neutral",
        confidence=0.4,
    )
    rows = db.fetch_analysis(user_id=user_id, limit=50)
    row = next(r for r in rows if r[0] == aid)
    assert "a@b.com" not in row[2]
    assert "example.com" not in row[3]
    assert row[8] == "web"