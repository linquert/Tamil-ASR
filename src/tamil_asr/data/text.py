from __future__ import annotations

import hashlib
import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>\n]{1,128}\|>")
_TAMIL_START = 0x0B80
_TAMIL_END = 0x0BFF


def normalize_transcript(text: object) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFC", str(text))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = _CONTROL_TOKEN_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def transcript_fingerprint(text: str) -> str:
    normalized = normalize_transcript(text).casefold()
    canonical = "".join(
        ch for ch in normalized if unicodedata.category(ch)[0] in {"L", "N", "M"}
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def tamil_fraction(text: str) -> float:
    significant = [ch for ch in text if ch.isalpha() or ch.isdigit()]
    if not significant:
        return 0.0
    tamil = sum(_TAMIL_START <= ord(ch) <= _TAMIL_END for ch in significant)
    return tamil / len(significant)


def stable_bucket(value: str, modulo: int = 10_000) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo
