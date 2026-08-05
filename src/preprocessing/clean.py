import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B50\u2190-\u21FF\u2B00-\u2BFF]"
)


def clean_text(
    text,
    remove_urls=True,
    remove_emails=True,
    remove_mentions=True,
    remove_hashtags=True,
    remove_emoji=False,
    collapse_whitespace=True,
    unicode_normalize="NFC",
):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if unicode_normalize:
        text = unicodedata.normalize(unicode_normalize, text)
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_emails:
        text = _EMAIL_RE.sub(" ", text)
    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)
    if remove_hashtags:
        text = _HASHTAG_RE.sub(" ", text)
    if remove_emoji:
        text = _EMOJI_RE.sub(" ", text)
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
