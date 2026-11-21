"""Tests for ⑤ VERIFY (sentineldesk/verify.py). Run: python -m pytest tests/test_verify.py -q"""
from __future__ import annotations

import time

import pytest

from sentineldesk.verify import UNSUPPORTED_ASK, VERIFY_FAILED, extract_values, verify

RETRIEVED = [
    {"ticket_id": "TKT-17145", "answer": "Please restart the VPN client and clear the cache. "
                                         "If it persists, call <tel_num> and quote <acc_num>."},
    {"ticket_id": "TKT-03282", "answer": "Refunds for double charges are processed within 24 hours. "
                                         "The fee of 1000.50 EUR will be credited, a 10% discount applies."},
    {"ticket_id": "TKT-05470", "answer": "Sehr geehrte/r <name>, wir prüfen die Abbuchung innerhalb von 3 Tagen."},
]
TICKET_EN = "VPN down\nOur VPN client fails since this morning on PHP 8.0 servers."
TICKET_DE = "Doppelte Abbuchung\nMir wurde die Gebühr zweimal abgebucht. Bitte prüfen."


def draft(reply, cited=("TKT-17145",), asks=()):
    return {"reply": reply, "cited_ids": list(cited), "unsupported_asks": list(asks)}


def failed(result):
    return {f["check"] for f in result["failures"]}


# ---------------------------------------------------------------- happy path
def test_clean_reply_passes():
    r = verify(draft("Thanks for reaching out. Please restart the VPN client and clear the cache [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]
    assert [c["check"] for c in r["checks"]] == ["format", "citations", "numbers", "placeholders",
                                                  "language", "unsupported_asks"]


# ---------------------------------------------------------------- citations
def test_unretrieved_citation_fails():
    r = verify(draft("Please restart the VPN client and clear the cache, see the earlier case.",
                     cited=["TKT-99999"]), RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"citations"}


def test_inline_unretrieved_citation_fails_even_if_field_is_clean():
    r = verify(draft("Please restart the VPN client and clear the cache [TKT-17145] [TKT-12345]."),
               RETRIEVED, TICKET_EN, "en")
    assert "citations" in failed(r)
    assert "TKT-12345" in r["failures"][0]["detail"]


def test_no_citation_fails():
    r = verify(draft("Please restart the VPN client and clear the cache today.", cited=[]),
               RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"citations"}


# ---------------------------------------------------------------- numbers
def test_invented_amount_fails():
    r = verify(draft("We will credit you €50 for the trouble with the VPN client [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}
    assert "€50" in r["failures"][0]["detail"]


def test_invented_duration_fails():
    r = verify(draft("Your refund for the double charge will arrive within 48 hours [TKT-03282].",
                     cited=["TKT-03282"]), RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}


def test_grounded_duration_passes_across_languages():
    # "24 hours" in the reply, "24 hours" in the source; and the German "3 Tagen" matches "3 days".
    r = verify(draft("Refunds for double charges are processed within 24 hours, review takes 3 days "
                     "[TKT-03282][TKT-05470].", cited=["TKT-03282", "TKT-05470"]), RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]


def test_number_word_before_unit_is_checked():
    r = verify(draft("The review of your double charge takes three days in total [TKT-05470].",
                     cited=["TKT-05470"]), RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]  # "three days" == "3 Tagen"
    r = verify(draft("The review of your double charge takes five days in total [TKT-05470].",
                     cited=["TKT-05470"]), RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}


def test_plain_number_word_is_not_a_claim():
    assert extract_values("One of our agents will help you.") == []


def test_german_number_format_matches_english_source():
    r = verify(draft("Die Gebühr von 1.000,50 € wird gutgeschrieben, es gilt ein Rabatt von 10 % [TKT-03282].",
                     cited=["TKT-03282"]), RETRIEVED, TICKET_DE, "de")
    assert "numbers" not in failed(r), r["failures"]


def test_currency_must_match_not_just_the_number():
    r = verify(draft("The fee of 1000.50 USD will be credited back to you [TKT-03282].", cited=["TKT-03282"]),
               RETRIEVED, TICKET_EN, "en")
    assert failed(r) == {"numbers"}  # the source says EUR


def test_number_from_customer_ticket_is_allowed():
    r = verify(draft("Please restart the VPN client on your PHP 8.0 servers and clear the cache [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]


# ---------------------------------------------------------------- placeholders and invented PII
def test_kept_placeholders_pass():
    r = verify(draft("If the VPN client still fails, call <tel_num> and quote <acc_num> [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert r["passed"], r["failures"]


def test_invented_phone_number_fails():
    r = verify(draft("If the VPN client still fails, call us on +49 30 1234567 any time [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert "placeholders" in failed(r)


def test_invented_email_fails():
    r = verify(draft("If the VPN client still fails, write to help@example.com for support [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert "placeholders" in failed(r)


def test_unknown_placeholder_fails():
    r = verify(draft("If the VPN client still fails, contact <manager_name> directly please [TKT-17145]."),
               RETRIEVED, TICKET_EN, "en")
    assert "placeholders" in failed(r)


# ---------------------------------------------------------------- language
def test_wrong_language_fails():
    r = verify(draft("Thank you, please restart the VPN client and clear the cache today [TKT-17145]."),
               RETRIEVED, TICKET_DE, "de")
    assert "language" in failed(r)


def test_short_reply_language_not_enforced():
    r = verify(draft("Please restart it [TKT-17145]."), RETRIEVED, TICKET_DE, "de")
    assert "language" not in failed(r)


def test_german_reply_to_german_ticket_passes():
    r = verify(draft("Sehr geehrte/r <name>, wir prüfen die Abbuchung innerhalb von 3 Tagen und melden uns "
                     "bei Ihnen [TKT-05470].", cited=["TKT-05470"]), RETRIEVED, TICKET_DE, "de")
    assert r["passed"], r["failures"]


# ---------------------------------------------------------------- unsupported asks
def test_unsupported_ask_fails_with_its_own_code():
    r = verify(draft("Please restart the VPN client and clear the cache [TKT-17145].",
                     asks=["refund guarantee for downtime"]), RETRIEVED, TICKET_EN, "en")
    assert r["failures"] == [{"check": "unsupported_asks", "code": UNSUPPORTED_ASK,
                              "detail": "model could not ground: ['refund guarantee for downtime']"}]


# ---------------------------------------------------------------- malformed input never crashes
@pytest.mark.parametrize("bad", [None, "just a string", {}, {"reply": ""}, {"reply": 42, "cited_ids": "TKT-17145"}])
def test_malformed_draft_fails_without_crashing(bad):
    r = verify(bad, RETRIEVED, TICKET_EN, "en")
    assert not r["passed"]
    assert "format" in failed(r)
    assert all(f["code"] == VERIFY_FAILED for f in r["failures"] if f["check"] == "format")


def test_empty_retrieval_fails_citations():
    r = verify(draft("Please restart the VPN client and clear the cache [TKT-17145]."), [], TICKET_EN, "en")
    assert "citations" in failed(r)


def test_too_long_reply_fails():
    r = verify(draft("word " * 300 + "[TKT-17145]"), RETRIEVED, TICKET_EN, "en")
    assert "format" in failed(r)


def test_all_failures_are_reported_not_just_the_first():
    r = verify(draft("Call +49 30 1234567, we credit €50 within 48 hours.", cited=["TKT-99999"],
                     asks=["compensation"]), RETRIEVED, TICKET_EN, "en")
    assert {"citations", "numbers", "placeholders", "unsupported_asks"} <= failed(r)


def test_fast_enough_for_the_live_path():
    d = draft("Refunds for double charges are processed within 24 hours [TKT-03282].", cited=["TKT-03282"])
    t = time.perf_counter()
    for _ in range(200):
        verify(d, RETRIEVED, TICKET_EN, "en")
    assert (time.perf_counter() - t) / 200 < 0.01  # under 10 ms per draft
