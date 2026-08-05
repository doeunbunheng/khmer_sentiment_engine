import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:0\d{1,2})?[\s.-]?\d{3}[\s.-]?\d{3}[\s.-]?\d{3,4}"
)
_KHMER_ID_RE = re.compile(r"[0-9]{1,2}[\s-]?[0-9]{3}[\s-]?[0-9]{3}")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_MENTION_RE = re.compile(r"@\w+")
_PLACEHOLDER = " [ANONYMIZED] "


def anonymize_text(text, names=None):
    """Replace PII (URLs, emails, phones, Khmer IDs, cards, names) with a placeholder.

    names: optional iterable of display names / usernames to mask (case-insensitive,
    whole-word only so ordinary words like 'may' are not affected).
    """
    if not text or not isinstance(text, str):
        return text
    for pattern in (_URL_RE, _EMAIL_RE, _PHONE_RE, _KHMER_ID_RE, _CREDIT_CARD_RE):
        text = pattern.sub(_PLACEHOLDER, text)
    text = _MENTION_RE.sub(_PLACEHOLDER, text)
    for name in names or []:
        name = str(name).strip()
        if name:
            text = re.sub(
                rf"\b{re.escape(name)}\b", _PLACEHOLDER, text, flags=re.IGNORECASE
            )
    return re.sub(r"\s+", " ", text).strip()


def has_pii(text, names=None):
    if not text or not isinstance(text, str):
        return False
    for pattern in (_URL_RE, _EMAIL_RE, _PHONE_RE, _KHMER_ID_RE, _CREDIT_CARD_RE, _MENTION_RE):
        if pattern.search(text):
            return True
    for name in names or []:
        name = str(name).strip()
        if name and re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            return True
    return False