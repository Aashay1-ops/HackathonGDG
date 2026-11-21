"""SentinelDesk: Hybrid customer support triage, retrieval, safety gate, and grounded drafting."""

from sentineldesk.language import (
    LanguagePrediction,
    detect_language,
    detect_language_batch,
    detect_language_with_confidence,
)

__all__ = [
    "LanguagePrediction",
    "detect_language",
    "detect_language_batch",
    "detect_language_with_confidence",
]
