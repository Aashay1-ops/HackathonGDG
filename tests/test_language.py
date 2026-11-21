"""Unit and benchmark tests for SentinelDesk language detection (`en` vs `de`)."""

from __future__ import annotations

import math
import time

import pytest

from sentineldesk.language import (
    LanguagePrediction,
    detect_language,
    detect_language_batch,
    detect_language_with_confidence,
)


class TestLanguageDetectionCore:
    """Verify correctness across English, German, short, and edge-case tickets."""

    @pytest.mark.parametrize(
        ("subject", "body", "expected"),
        [
            (
                "Account Disruption",
                "Dear Customer Support Team, I am writing to report a significant problem with the centralized account management portal.",
                "en",
            ),
            (
                "Query About Smart Home System Integration Features",
                "I hope this message reaches you well. I am reaching out to request detailed information about the capabilities of your smart home integration products.",
                "en",
            ),
            (
                "VPN is down",
                "Our hospital staff cannot access patient records since this morning.",
                "en",
            ),
            (
                None,
                "Please help me reset my billing password on the portal.",
                "en",
            ),
            (
                "Wesentlicher Sicherheitsvorfall",
                "Sehr geehrtes Support-Team, ich möchte einen gravierenden Sicherheitsvorfall melden, der gegenwärtig mehrere Komponenten unserer Infrastruktur betrifft.",
                "de",
            ),
            (
                "Sicherheitsverletzung gemeldet",
                "Es wurde eine Sicherheitsverletzung im IT-System festgestellt.",
                "de",
            ),
            (
                "Probleme mit dem Smart-Türklingel",
                "Ich beschreibe das Problem mit meinem Smart-Türklingel-Gerät. Leider gibt es einen Zusammenbruch der Verbindung.",
                "de",
            ),
            (
                None,
                "Kein Zugriff auf das System nach dem letzten Update. Bitte helfen Sie uns dringend.",
                "de",
            ),
            (
                "Störung",
                "Server ausgefallen",
                "de",
            ),
        ],
    )
    def test_standard_en_and_de_tickets(
        self, subject: str | None, body: str, expected: str
    ) -> None:
        pred = detect_language_with_confidence(subject, body)
        assert isinstance(pred, LanguagePrediction)
        assert pred.language == expected
        assert 0.5 < pred.confidence <= 1.0
        assert detect_language(subject, body) == expected

    @pytest.mark.parametrize(
        ("subject", "body"),
        [
            (None, None),
            ("", ""),
            ("   ", "\n\t "),
            (float("nan"), float("nan")),
            (float("nan"), ""),
            (None, float("nan")),
            (12345, None),
            ("!!!", "??? @@@ ### 123-456"),
        ],
    )
    def test_missing_or_empty_inputs_never_crash(
        self, subject: object, body: object
    ) -> None:
        """Per README: `subject` can be empty/NaN, so never crash on it."""
        pred = detect_language_with_confidence(subject, body)
        assert pred.language in ("en", "de")
        assert 0.5 <= pred.confidence <= 1.0
        assert not math.isnan(pred.confidence)

    def test_umlaut_only_fallback(self) -> None:
        pred = detect_language_with_confidence("Überprüfung", "")
        assert pred.language == "de"
        assert pred.confidence >= 0.7

    def test_morphological_suffix_tie_breaker(self) -> None:
        # No stopwords or umlauts, only German morphological suffixes (-ung, -keit)
        pred_de = detect_language_with_confidence(
            "Systemaktualisierung", "Verfügbarkeitsprüfung fehlgeschlagen"
        )
        assert pred_de.language == "de"

        # No stopwords, only English morphological suffixes (-tion, -ment, -ing)
        pred_en = detect_language_with_confidence(
            "Configuration", "Deployment processing failed"
        )
        assert pred_en.language == "en"

    def test_batch_detection(self) -> None:
        subjects = ["VPN outage", "Wesentlicher Sicherheitsvorfall", None]
        bodies = [
            "We are experiencing a critical network failure.",
            "Sehr geehrtes Team, der Server ist ausgefallen.",
            "Thank you for your help with our billing account.",
        ]
        results = detect_language_batch(subjects, bodies)
        assert results == ["en", "de", "en"]

    def test_latency_benchmark(self) -> None:
        """Ensure single-ticket inference is under 0.1 ms (target ~0.02 ms)."""
        subj = "Account Disruption"
        body = (
            "Dear Customer Support Team, I am writing to report a significant "
            "problem with the centralized account management portal."
        )
        n_iters = 5000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            detect_language(subj, body)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms = elapsed_ms / n_iters
        assert avg_ms < 0.1, f"Expected < 0.1ms per ticket, got {avg_ms:.4f}ms"
