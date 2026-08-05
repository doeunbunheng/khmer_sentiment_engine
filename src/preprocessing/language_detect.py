import re

_KHMER_RE = re.compile(r"[\u1780-\u17FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text):
    if text is None or not text.strip():
        return "unknown"
    has_khmer = bool(_KHMER_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_khmer and has_latin:
        return "mixed"
    if has_khmer:
        return "khmer"
    if has_latin:
        return "english"
    return "unknown"
