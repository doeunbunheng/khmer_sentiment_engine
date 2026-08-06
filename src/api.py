"""FastAPI server — serves the production prediction pipeline, hardened.

Endpoints:
  GET  /health             -> model + DB status (no auth)
  POST /auth/login         -> {username, password} -> signed token (rate-limited + lockout)
  POST /predict            -> Bearer token required; {text, user_id?, consent?, names?}
  GET  /feedback           -> Bearer token + Admin role required; latest feedback rows

Security layers:
  1. HMAC-signed bearer tokens (src/common/security.py), 24h expiry, secret in .env
  2. Rate limits: /auth/login 5/min/IP, /predict 120/min/IP (env-tunable)
  3. Login lockout: 5 failed logins -> 15-min block
  4. Input caps: text <= 2000 chars (reject 422), username/password caps
  5. Admin-only /feedback (403 for non-admins)
  6. CORS locked to ALLOWED_ORIGINS env (never *)
  7. No PII logged: nothing here logs request bodies or tokens
  8. HTTPS: for non-localhost, run behind a TLS proxy or use
     `uvicorn src.api:app --ssl-keyfile key.pem --ssl-certfile cert.pem`

Run locally:
  .venv\\Scripts\\uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

import os
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.common import db as common_db
from src.common.security import (
    create_token,
    login_blocked,
    register_login_attempt,
    verify_token,
)
from src.predict import predict_and_save

LOGIN_LIMIT = os.getenv("API_LOGIN_LIMIT", "5/minute")
PREDICT_LIMIT = os.getenv("API_PREDICT_LIMIT", "120/minute")
FEEDBACK_LIMIT = os.getenv("API_FEEDBACK_LIMIT", "30/minute")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Khmer Sentiment Engine",
    description="3-class sentiment + business aspects + emotions for Khmer/English/mixed text.",
    version="1.1.0",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- auth -------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    token: str
    user_id: int
    role: str
    expires_in_seconds: int


@app.post("/auth/login", response_model=LoginResponse)
@limiter.limit(LOGIN_LIMIT)
def login(request: Request, body: LoginRequest):
    if login_blocked(body.username):
        raise HTTPException(
            status_code=429,
            detail="too many failed logins — account temporarily locked",
        )
    res = common_db.login_user(body.username, body.password)
    ok = bool(res.get("ok"))
    blocked, _ = register_login_attempt(body.username, ok)
    if blocked:
        raise HTTPException(
            status_code=429,
            detail="too many failed logins — account temporarily locked",
        )
    if not ok:
        raise HTTPException(status_code=401, detail="invalid credentials")
    from src.common.security import TOKEN_TTL_SECONDS

    return LoginResponse(
        token=create_token(res["user_id"], res["role"]),
        user_id=res["user_id"],
        role=res["role"],
        expires_in_seconds=TOKEN_TTL_SECONDS,
    )


_bearer = HTTPBearer(auto_error=False)


def _bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
):
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    user = verify_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user


def _require_admin(user: dict = Depends(_bearer_token)):
    if user["role"] != "Admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return user


# ---- predict ----------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=0, max_length=2000)
    user_id: Optional[int] = None
    consent: bool = False
    names: Optional[List[str]] = Field(default=None, max_length=50)


class PredictResponse(BaseModel):
    text: str
    language: str
    sentiment: str
    confidence: float
    uncertain: bool = Field(
        default=False,
        description="True when confidence is below model.uncertainty_threshold "
        "(default 0.90) — treat as 'ask the user' instead of a guess",
    )
    translated_text: Optional[str] = None
    aspects: dict
    saved_id: Optional[int] = None
    saved_error: Optional[bool] = None


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(PREDICT_LIMIT)
def predict(request: Request, req: PredictRequest, user: dict = Depends(_bearer_token)):
    return predict_and_save(
        text=req.text,
        user_id=req.user_id,
        consent=req.consent,
        names=req.names,
    )


# ---- feedback (admin only) --------------------------------------------------

class FeedbackRow(BaseModel):
    id: int
    user_id: int
    consent_granted: bool
    lang_detect: str
    sentiment: str
    confidence: float
    aspects: dict


@app.get("/feedback", response_model=List[FeedbackRow])
@limiter.limit(FEEDBACK_LIMIT)
def feedback(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    admin: dict = Depends(_require_admin),
):
    rows = common_db.fetch_feedback(limit=limit)
    return [
        FeedbackRow(
            id=r[0],
            user_id=r[1],
            consent_granted=r[2],
            lang_detect=r[5],
            sentiment=r[7],
            confidence=r[8],
            aspects=r[9],
        )
        for r in rows
    ]


# ---- health (open) ----------------------------------------------------------

@app.get("/health")
def health():
    db_ok = True
    try:
        common_db.connect().close()
    except Exception:
        db_ok = False
    return {"status": "ok", "model": "khmer-sentiment-3class-v2", "db": db_ok}
