import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json

from src.common.config import DB
from src.preprocessing.anonymize import anonymize_text, has_pii
from src.preprocessing.clean import clean_text

load_dotenv()


def connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", DB["host"]),
        port=int(os.getenv("DB_PORT", DB["port"])),
        dbname=os.getenv("DB_NAME", DB["name"]),
        user=os.getenv("DB_USER", DB["user"]),
        password=os.getenv("DB_PASSWORD", DB["password"]),
    )


def register_user(
    username, password, role="User", full_name=None, email=None, phone=None
):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM register_user(%s, %s, %s, %s, %s, %s)",
                (username, password, role, full_name, email, phone),
            )
            row = cur.fetchone()
    return row[0] if row else None


def login_user(username, password):
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM login_user(%s, %s)", (username, password))
            row = cur.fetchone()
    if not row:
        return {"ok": False}
    user_id, role, ok = row
    return {"user_id": user_id, "role": role, "ok": bool(ok)}


def save_feedback(
    user_id,
    text_orig,
    lang_detect,
    sentiment,
    confidence,
    text_translated=None,
    aspects=None,
    consent=False,
    names=None,
    text_anonymized=None,
):
    if not consent:
        raise ValueError("consent not granted; feedback not stored")
    if text_anonymized is None:
        text_anonymized = anonymize_text(text_orig, names=names)
    if not text_anonymized or not text_anonymized.strip():
        raise ValueError("empty anonymized text; nothing to store")
    if aspects is None:
        aspects = {}
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_feedback
                    (user_id, consent_granted, text_orig, text_anonymized,
                     lang_detect, text_translated, sentiment, confidence, aspects)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_id,
                    consent,
                    text_orig,
                    text_anonymized,
                    lang_detect,
                    text_translated,
                    sentiment,
                    float(confidence),
                    Json(aspects),
                ),
            )
            return cur.fetchone()[0]


def fetch_feedback(user_id=None, limit=100):
    query = "SELECT * FROM user_feedback"
    params = []
    if user_id is not None:
        query += " WHERE user_id = %s"
        params.append(user_id)
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def log_analysis(
    user_id,
    text,
    language,
    sentiment,
    confidence,
    aspects=None,
    source="web",
):
    """Insert an anonymized prediction record into analysis_records (audit log).

    The text is anonymized before storage, so the audit log never keeps PII
    and needs no user consent (unlike user_feedback).
    """
    text_anon = anonymize_text(text)
    text_clean = clean_text(text_anon)
    if aspects is None:
        aspects = {}
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analysis_records
                    (user_id, text, text_cleaned, language, sentiment,
                     confidence, aspects, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_id,
                    text_anon,
                    text_clean,
                    language,
                    sentiment,
                    float(confidence),
                    Json(aspects),
                    source,
                ),
            )
            return cur.fetchone()[0]


def fetch_analysis(user_id=None, limit=100):
    query = (
        "SELECT id, user_id, text, text_cleaned, language, sentiment, "
        "confidence, aspects, source, created_at FROM analysis_records"
    )
    params = []
    if user_id is not None:
        query += " WHERE user_id = %s"
        params.append(user_id)
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()