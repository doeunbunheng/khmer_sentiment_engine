import pytest

from src.common import db

try:
    db.connect().close()
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DB_AVAILABLE, reason="PostgreSQL not reachable — start the DB first"
)

_UNIQUE = "test_auth_1"


@pytest.fixture(scope="module", autouse=True)
def registered_user():
    try:
        user_id = db.register_user(_UNIQUE, "Pass123!", "User", None, f"{_UNIQUE}@x.kh", "010-999-001")
    except Exception:
        user_id = db.login_user(_UNIQUE, "Pass123!")["user_id"]
    yield user_id


def test_register_returns_id(registered_user):
    assert registered_user is not None


def test_duplicate_username_rejected(registered_user):
    with pytest.raises(Exception):
        db.register_user(_UNIQUE, "Other1!", "User", None, "other@x.kh", "011-999-002")


def test_login_ok(registered_user):
    assert db.login_user(_UNIQUE, "Pass123!")["ok"] is True


def test_login_wrong_password(registered_user):
    assert db.login_user(_UNIQUE, "wrongpass")["ok"] is False