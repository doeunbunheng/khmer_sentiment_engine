from src.preprocessing.anonymize import anonymize_text, has_pii


def test_phone_masked():
    out = anonymize_text("Call 012-345-678 now")
    assert "012-345-678" not in out
    assert "[ANONYMIZED]" in out


def test_email_fully_masked():
    out = anonymize_text("Contact a@b.com please")
    assert "a@b.com" not in out
    assert out.count("[ANONYMIZED]") == 1


def test_email_mention_order():
    out = anonymize_text("reach @me at a@b.com ok")
    assert "a@b.com" not in out
    assert "@me" not in out


def test_url_masked():
    out = anonymize_text("see https://example.com/x now")
    assert "example.com" not in out
    assert "[ANONYMIZED]" in out


def test_mention_masked():
    out = anonymize_text("hey @user how are you")
    assert "@user" not in out


def test_plain_text_unchanged():
    assert anonymize_text("ផលិតផលល្អណាស់") == "ផលិតផលល្អណាស់"
    assert anonymize_text("just words") == "just words"


def test_names_masked():
    out = anonymize_text("Sok bought this", names=["Sok"])
    assert "Sok" not in out
    assert "[ANONYMIZED]" in out


def test_name_not_partial_word():
    out = anonymize_text("soccer is fun", names=["sok"])
    assert out == "soccer is fun"


def test_has_pii_true():
    assert has_pii("call 099 999 999")
    assert has_pii("mail a@b.com")


def test_has_pii_false():
    assert not has_pii("hello world")
    assert not has_pii("វាល្អណាស់")