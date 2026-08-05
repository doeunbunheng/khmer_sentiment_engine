import re

_KHMER_RE = re.compile(r"[\u1780-\u17FF]+")
_WHITESPACE_RE = re.compile(r"\s+")

try:
    from khmernltk import word_tokenize

    _HAVE_KHMERNLTK = True
except ImportError:
    _HAVE_KHMERNLTK = False


def segment_text(text):
    if not text or not text.strip():
        return ""
    if _HAVE_KHMERNLTK:
        try:
            return _WHITESPACE_RE.sub(" ", " ".join(word_tokenize(text))).strip()
        except Exception:
            pass
    return _regex_fallback(text)


def _regex_fallback(text):
    return " ".join(text.split())
