"""API security helpers — signed tokens, login lockout.

- `create_token()` / `verify_token()`: HMAC-SHA256 signed tokens
  (stdlib only, no JWT dependency). Payload: user_id, role, expiry, nonce.
  Secret from `API_SECRET` env var.
- Login lockout: 5 failed logins per username -> 15-minute block (in-memory;
  fine for the single-process local deployment).

Never log tokens, passwords, or raw text anywhere in this module.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

TOKEN_TTL_SECONDS = int(os.getenv("API_TOKEN_TTL", "86400"))  # 24h
MAX_LOGIN_ATTEMPTS = int(os.getenv("API_MAX_LOGIN_ATTEMPTS", "5"))
LOCKOUT_SECONDS = int(os.getenv("API_LOCKOUT_SECONDS", "900"))  # 15 min


def _secret():
    secret = os.getenv("API_SECRET", "")
    if not secret:
        raise RuntimeError("API_SECRET env var is not set — set it in .env")
    return secret.encode()


def create_token(user_id, role, ttl_seconds=TOKEN_TTL_SECONDS):
    """Return a signed token: base64(payload).hmac_sha256_hex."""
    payload = {
        "uid": int(user_id),
        "role": role,
        "exp": int(time.time()) + ttl_seconds,
        "nonce": secrets.token_hex(8),
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    sig = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify_token(token):
    """Return {user_id, role} or None (invalid/expired/tampered)."""
    try:
        body, sig = token.rsplit(".", 1)
        expected = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
        if int(payload.get("exp", 0)) < time.time():
            return None
        return {"user_id": int(payload["uid"]), "role": payload.get("role", "User")}
    except Exception:
        return None


# ---- login lockout (in-memory, single process) ------------------------------

_attempts = {}
_blocked_until = {}


def reset_lockout(username=None):
    if username is None:
        _attempts.clear()
        _blocked_until.clear()
    else:
        _attempts.pop(username, None)
        _blocked_until.pop(username, None)


def login_blocked(username):
    """True if the username is currently locked out."""
    return _blocked_until.get(username, 0) > time.time()


def register_login_attempt(username, ok):
    """Record a login result; returns (blocked, seconds_remaining).

    A failed login increments the counter; on the 5th failure the username is
    blocked for 15 minutes. A successful login clears the counters.
    """
    now = time.time()
    if ok:
        _attempts.pop(username, None)
        _blocked_until.pop(username, None)
        return False, 0
    count = _attempts.get(username, 0) + 1
    _attempts[username] = count
    if count >= MAX_LOGIN_ATTEMPTS:
        _blocked_until[username] = now + LOCKOUT_SECONDS
        _attempts[username] = 0
        return True, LOCKOUT_SECONDS
    return False, 0
