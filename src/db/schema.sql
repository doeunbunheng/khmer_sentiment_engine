-- ============================================================
--  KHMER SENTIMENT ENGINE  --  PostgreSQL 15+
--  Roles: Admin | User        (adapts the Oracle store example)
--  Security: passwords hashed in-DB with pgcrypto bcrypt
-- ============================================================

-- STEP 0: EXTENSIONS
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- STEP 1: CREATE TABLES
-- ============================================================

-- Secure accounts (login/register) — plain passwords never stored
CREATE TABLE IF NOT EXISTS users (
    id              BIGSERIAL PRIMARY KEY,
    username        VARCHAR(50) UNIQUE NOT NULL,
    password_hash   VARCHAR(100) NOT NULL,  -- crypt(pwd, gen_salt('bf'))
    role            VARCHAR(20) NOT NULL DEFAULT 'User'
                    CHECK (role IN ('Admin','User')),
    consent_granted BOOLEAN NOT NULL DEFAULT FALSE,
    consent_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login      TIMESTAMPTZ
);

-- User PII lives here, SEPARATE from feedback
CREATE TABLE IF NOT EXISTS user_profiles (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id),
    full_name   VARCHAR(100),
    email       VARCHAR(100) UNIQUE,
    phone       VARCHAR(20),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Week 2 feedback + Week 3: user link, consent, anonymized copy
CREATE TABLE IF NOT EXISTS user_feedback (
    id                BIGSERIAL PRIMARY KEY,
    user_id           BIGINT REFERENCES users(id),
    consent_granted   BOOLEAN NOT NULL DEFAULT FALSE,
    text_orig         TEXT NOT NULL,
    text_anonymized   TEXT NOT NULL,          -- PII stripped BEFORE save (never NULL)
    lang_detect       VARCHAR(10) NOT NULL,
    text_translated   TEXT,
    sentiment         VARCHAR(10) NOT NULL,
    confidence        REAL NOT NULL,
    aspects           JSONB NOT NULL DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Auto/offline analysis records (keep as-is, now optionally linked)
CREATE TABLE IF NOT EXISTS analysis_records (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    text TEXT NOT NULL,
    text_cleaned TEXT,
    language VARCHAR(20) DEFAULT 'khmer',
    sentiment VARCHAR(20) NOT NULL,
    confidence REAL NOT NULL,
    topics JSONB NOT NULL DEFAULT '{}',
    aspects JSONB NOT NULL DEFAULT '{}',
    source VARCHAR(50) DEFAULT 'web',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- STEP 2: TRIGGERS (PostgreSQL ~ your Oracle Step 4)
-- ============================================================

-- Auto-set updated_at on any user row change
CREATE OR REPLACE FUNCTION fn_set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

-- Record consent timestamp the moment consent is granted (= your consent audit)
CREATE OR REPLACE FUNCTION fn_set_consent_at() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.consent_granted AND (OLD.consent_granted IS DISTINCT FROM TRUE OR OLD.consent_at IS NULL) THEN
        NEW.consent_at := now();
    END IF;
    IF NOT NEW.consent_granted THEN
        NEW.consent_at := NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_users_consent ON users;
CREATE TRIGGER trg_users_consent
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION fn_set_consent_at();

-- Security gate: refuse to save feedback without consent or an anonymized copy
CREATE OR REPLACE FUNCTION fn_require_consent() RETURNS TRIGGER AS $$
BEGIN
    IF NOT NEW.consent_granted THEN
        RAISE EXCEPTION 'consent not granted; feedback not stored';
    END IF;
    IF NEW.text_anonymized IS NULL OR NEW.text_anonymized = '' THEN
        RAISE EXCEPTION 'anonymized text required before storing feedback';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_feedback_consent ON user_feedback;
CREATE TRIGGER trg_feedback_consent
    BEFORE INSERT ON user_feedback
    FOR EACH ROW EXECUTE FUNCTION fn_require_consent();

-- ============================================================
-- STEP 3: PROCEDURES / FUNCTIONS (like Oracle proc_register_customer)
-- ============================================================

-- Secure registration: hashes with bcrypt, rejects duplicate username/email/phone
CREATE OR REPLACE FUNCTION register_user(
    p_username  VARCHAR,
    p_password  VARCHAR,
    p_role      VARCHAR DEFAULT 'User',
    p_full_name VARCHAR DEFAULT NULL,
    p_email     VARCHAR DEFAULT NULL,
    p_phone     VARCHAR DEFAULT NULL
) RETURNS TABLE (user_id BIGINT, out_role VARCHAR) AS $$
DECLARE
    v_user_id BIGINT;
BEGIN
    IF (SELECT 1 FROM user_profiles WHERE email = p_email) IS NOT NULL THEN
        RAISE EXCEPTION 'Email already registered!';
    END IF;
    IF EXISTS (SELECT 1 FROM user_profiles WHERE phone = p_phone AND p_phone IS NOT NULL) THEN
        RAISE EXCEPTION 'Phone already registered!';
    END IF;
    IF EXISTS (SELECT 1 FROM users WHERE username = p_username) THEN
        RAISE EXCEPTION 'Username already taken!';
    END IF;

    INSERT INTO users (username, password_hash, role)
    VALUES (p_username, crypt(p_password, gen_salt('bf')), p_role)
    RETURNING id INTO v_user_id;

    INSERT INTO user_profiles (user_id, full_name, email, phone)
    VALUES (v_user_id, p_full_name, p_email, p_phone);

    RETURN QUERY SELECT v_user_id, p_role;
END;
$$ LANGUAGE plpgsql;

-- Login: checks bcrypt hash, updates last_login
CREATE OR REPLACE FUNCTION login_user(
    p_username VARCHAR,
    p_password VARCHAR
) RETURNS TABLE (out_user_id BIGINT, out_role VARCHAR, out_ok BOOLEAN) AS $$
DECLARE
    v_user users%ROWTYPE;
BEGIN
    SELECT * INTO v_user FROM users WHERE username = p_username;
    IF NOT FOUND THEN
        RETURN QUERY SELECT NULL::BIGINT, NULL::VARCHAR, FALSE;
        RETURN;
    END IF;

    IF v_user.password_hash = crypt(p_password, v_user.password_hash) THEN
        UPDATE users SET last_login = now() WHERE id = v_user.id;
        RETURN QUERY SELECT v_user.id, v_user.role, TRUE;
    ELSE
        RETURN QUERY SELECT v_user.id, v_user.role, FALSE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- STEP 4: VIEWS (like Oracle Step 3)
-- ============================================================

-- What an Admin sees: registered users + consent status
CREATE OR REPLACE VIEW v_registered_users AS
SELECT
    u.id            AS user_id,
    u.username,
    u.role,
    u.consent_granted,
    u.consent_at,
    u.created_at    AS register_date,
    p.full_name,
    p.email,
    p.phone,
    u.last_login
FROM users u
LEFT JOIN user_profiles p ON p.user_id = u.id
ORDER BY u.created_at DESC;

-- Admin report: every stored prediction joined to user (anonymized text only)
CREATE OR REPLACE VIEW v_feedback_report AS
SELECT
    f.id,
    u.username,
    u.role,
    f.consent_granted,
    f.text_anonymized AS text,
    f.lang_detect,
    f.sentiment,
    f.confidence,
    f.aspects,
    f.created_at
FROM user_feedback f
LEFT JOIN users u ON u.id = f.user_id
ORDER BY f.created_at DESC;

-- Consent / usage stats
CREATE OR REPLACE VIEW v_feedback_stats AS
SELECT
    COUNT(*)                                          AS total_feedback,
    COUNT(*) FILTER (WHERE consent_granted)           AS with_consent,
    COUNT(*) FILTER (WHERE NOT consent_granted)       AS without_consent,
    f.sentiment,
    COUNT(*)                                          AS per_sentiment
FROM user_feedback f
GROUP BY sentiment;

-- ============================================================
-- STEP 5: INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_analysis_created ON analysis_records (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON user_feedback (created_at);
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_profiles_email ON user_profiles (email);

-- ============================================================
-- STEP 6: ROLES + GRANTS (like Oracle Steps 6–7)
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_admin') THEN
        CREATE ROLE role_admin;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_user') THEN
        CREATE ROLE role_user;
    END IF;
END
$$;

-- Admin: full control
GRANT SELECT, INSERT, UPDATE, DELETE ON users, user_profiles, user_feedback, analysis_records TO role_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO role_admin;
GRANT SELECT ON v_registered_users, v_feedback_report, v_feedback_stats TO role_admin;

-- User: only own data
GRANT SELECT ON users TO role_user;
GRANT SELECT, INSERT, UPDATE ON user_profiles TO role_user;
GRANT SELECT, INSERT ON user_feedback TO role_user;
GRANT SELECT ON v_feedback_report TO role_user;

-- ============================================================
-- STEP 7: SAMPLE DATA (optional test)
-- ============================================================

SELECT register_user('demo_admin', '132336BV132336', 'Admin', 'Admin Demo', 'admin@demo.kh', '010-000-000');
SELECT register_user('demo_user',  'user@132123', 'User',  'User Demo',  'user@demo.kh',  '012-000-000');

COMMIT;