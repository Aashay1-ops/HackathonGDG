"""The one text builder used by training, evaluation and the live demo."""
from __future__ import annotations

import math
import unicodedata


def _clean(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def build_text(subject, body) -> str:
    """subject (may be None/NaN/empty) + body -> one NFKC-normalized string. Never returns "nan"."""
    s, b = _clean(subject), _clean(body)
    text = normalize((s.strip() + "\n" + b.strip()).strip())
    if not text:
        raise ValueError("Ticket has no subject and no body")
    return text


def build_texts(subjects, bodies) -> list[str]:
    return [build_text(s, b) for s, b in zip(subjects, bodies)]
