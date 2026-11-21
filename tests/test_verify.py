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


# ================================================================ regressions from the cold review of PR #5
def one(answer, language=None, tid="TKT-00001"):
    row = {"ticket_id": tid, "answer": answer}
    if language:
        row["language"] = language
    return [row]


def check(reply, answer, lang="en", ticket="Please help with my account.", language=None):
    return verify({"reply": reply, "cited_ids": ["TKT-00001"], "unsupported_asks": []},
                  one(answer, language), ticket, lang)


# 1. units must match: a number with an unknown or different unit can't borrow a bare number from a source
@pytest.mark.parametrize("reply,answer,lang", [
    ("Your device comes with a 3 years warranty from us [TKT-00001].", "Wir prüfen das innerhalb von 3 Tagen.", "en"),
    ("Your request is processed in 10 seconds by our team [TKT-00001].", "A 10% discount applies to renewals.", "en"),
    ("Your plan is guaranteed for 24 more months from today [TKT-00001].", "Requests are handled within 24 hours.", "en"),
    ("We answer within 10 business hours on weekdays [TKT-00001].", "We answer within 10 days on weekdays.", "en"),
    ("Sie erhalten 10 Jahre lang einen Rabatt auf alles [TKT-00001].", "Es gilt ein Rabatt von 10 %.", "de"),
    ("You will be credited 10 times the monthly fee [TKT-00001].", "A 10% discount applies to renewals.", "en"),
])
def test_unit_mismatch_fails(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


def test_filler_word_between_number_and_unit_still_matches():
    r = check("We review your case within 3 business days at most [TKT-00001].", "Wir prüfen das innerhalb von 3 Tagen.")
    assert "numbers" not in failed(r), r["failures"]


# 2. number words: compounds read correctly, anything unreadable fails
@pytest.mark.parametrize("reply,answer,lang", [
    ("Your refund arrives within fifteen days from today [TKT-00001].", "Refunds take 3 days.", "en"),
    ("You get a fifty percent discount on the plan [TKT-00001].", "A 10% discount applies.", "en"),
    ("Sie erhalten zwanzig Euro als Gutschrift von uns [TKT-00001].", "Sie erhalten 10 Euro Gutschrift.", "de"),
    ("We will credit you a hundred euros for this [TKT-00001].", "We credit 10 euros.", "en"),
    ("Die Prüfung dauert eine Woche bei uns im Team [TKT-00001].", "Die Prüfung dauert 3 Tage.", "de"),
    ("Your refund arrives within thirty-three days from today [TKT-00001].", "Wir prüfen das innerhalb von 3 Tagen.", "en"),
])
def test_invented_number_words_fail(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


@pytest.mark.parametrize("reply,answer,lang", [
    ("Requests are handled within twenty-four hours by our team [TKT-00001].", "Requests are handled within 24 hours.", "en"),
    ("Anfragen werden innerhalb von vierundzwanzig Stunden bearbeitet [TKT-00001].", "Requests take 24 hours.", "de"),
])
def test_compound_number_words_match(reply, answer, lang):
    r = check(reply, answer, lang)
    assert "numbers" not in failed(r), r["failures"]


# 3. a separator is read by language: no 1000x amounts
@pytest.mark.parametrize("reply,answer,lang", [
    ("The plan costs €2,500 per user per month for you [TKT-00001].", "The plan costs €2.50 per user.", "en"),
    ("We add a credit of €1,000 to your account today [TKT-00001].", "A credit of €1 is added.", "en"),
    ("Wir schreiben Ihnen 1.000 € auf Ihrem Konto gut [TKT-00001].", "A credit of €1 is added.", "de"),
])
def test_amount_scale_mismatch_fails(reply, answer, lang):
    assert "numbers" in failed(check(reply, answer, lang))


def test_same_amount_in_both_formats_matches():
    r = check("Wir schreiben Ihnen 1.000 € auf Ihrem Konto gut [TKT-00001].", "A credit of €1,000 is added.", "de",
              language="en")
    assert "numbers" not in failed(r), r["failures"]


# 4. an invented email can't hide behind an existing <email> placeholder
def test_invented_email_next_to_placeholder_fails():
    r = check("Please write to <email> or directly to billing@contoso.com for help [TKT-00001].",
              "Please write to <email> for help.")
    assert "placeholders" in failed(r)


# 5. huge numbers never crash
@pytest.mark.parametrize("reply", ["€" + "1" * 400 + ",5 credit for you today [TKT-00001].",
                                   "Reference " + "9" * 4400 + " for you [TKT-00001]."])
def test_huge_numbers_fail_without_crashing(reply):
    r = check(reply, "A credit of €1 is added.")
    assert not r["passed"]


# 6. a German greeting can't carry an English reply past the language check
def test_mixed_language_reply_fails():
    r = check("Sehr geehrte/r <name>, vielen Dank für Ihre Nachricht. Please restart the VPN client and clear the "
              "cache, then try again [TKT-00001].", "Sehr geehrte/r <name>, bitte VPN neu starten.", "de")
    assert "language" in failed(r)


# 7. figures quoted verbatim from a source are not flagged as invented phone numbers
@pytest.mark.parametrize("reply,answer,lang", [
    ("Das Limit für Ihr Konto liegt bei 1.000.000 EUR pro Jahr [TKT-00001].",
     "Das Limit für Ihr Konto liegt bei 1.000.000 EUR pro Jahr.", "de"),
    ("Die Wartung beginnt am 21.11.2026 08:00 und dauert kurz [TKT-00001].",
     "Die Wartung beginnt am 21.11.2026 08:00.", "de"),
])
def test_grounded_figures_are_not_pii(reply, answer, lang):
    r = check(reply, answer, lang)
    assert "placeholders" not in failed(r), r["failures"]
    assert "numbers" not in failed(r), r["failures"]


# 8. odd inputs never crash; rows without a ticket_id can't be cited
def test_bad_retrieved_rows_and_ticket_types_do_not_crash():
    rows = [None, "x", {"answer": "no id here"}, {"ticket_id": "TKT-00001", "answer": "Restart the VPN client."}]
    r = verify({"reply": "Please restart the VPN client now [TKT-00001].", "cited_ids": ["TKT-00001"],
                "unsupported_asks": []}, rows, 12345, "en")
    assert r["passed"], r["failures"]


def test_row_without_ticket_id_cannot_be_cited():
    r = verify({"reply": "Please restart the VPN client now.", "cited_ids": ["None"], "unsupported_asks": []},
               [{"answer": "Restart the VPN client."}], "ticket", "en")
    assert "citations" in failed(r)


def test_uppercase_lang_is_accepted():
    r = check("Please restart the VPN client and clear the cache today [TKT-00001].", "Restart the VPN client.", "EN")
    assert "language" not in failed(r), r["failures"]


# 9. the same date in another format matches
def test_date_formats_match_across_languages():
    r = check("Die Wartung findet am 21.11.2026 um 10:30 statt, danach geht es weiter [TKT-00001].",
              "Maintenance is on 2026-11-21 at 10:30.", "de", language="en")
    assert "numbers" not in failed(r), r["failures"]


# 10. a huge one-word reply fails fast
def test_huge_single_token_reply_is_fast():
    t = time.perf_counter()
    r = check("1-" * 20000 + " [TKT-00001]", "Restart the VPN client.")
    assert "format" in failed(r)
    assert time.perf_counter() - t < 1.0


# 11. uppercase placeholders, invented links and spelled-out emails are caught; quoted links are fine
@pytest.mark.parametrize("reply", [
    "Please contact <Manager> about the VPN client issue [TKT-00001].",
    "Please download the fix from https://contoso-fix.example/vpn today [TKT-00001].",
    "Please write to help (at) contoso (dot) com about this [TKT-00001].",
])
def test_invented_placeholders_links_emails_fail(reply):
    assert "placeholders" in failed(check(reply, "Restart the VPN client."))


def test_link_quoted_from_source_passes():
    r = check("The guide is at https://docs.example.com/vpn for you [TKT-00001].",
              "See the guide at https://docs.example.com/vpn for steps.")
    assert "placeholders" not in failed(r), r["failures"]


# 12. the customer's own amount is not enough to promise it; plain numbers from the ticket still are
def test_amount_from_ticket_alone_cannot_be_promised():
    r = check("We will refund the €500 to your account shortly [TKT-00001].", "We review refund requests.",
              ticket="I was charged €500 twice, please refund.")
    assert "numbers" in failed(r)


# second pass, from re-running the reviewer's probes
@pytest.mark.parametrize("reply", ["A credit of €1.000 has been issued to your account today [TKT-00001].",
                                   "A credit of 1 000 € has been issued to your account today [TKT-00001]."])
def test_three_decimal_amounts_are_thousands(reply):
    assert "numbers" in failed(check(reply, "A credit of €1 is added."))


def test_space_separated_thousands_match():
    r = check("A credit of 1 000 € has been issued to your account today [TKT-00001].",
              "A credit of €1,000 is added.")
    assert "numbers" not in failed(r), r["failures"]


def test_leak_evidence_shows_the_whole_address():
    r = check("Please write to <email> or directly to billing@contoso.com for help [TKT-00001].",
              "Please write to <email> for help.")
    assert "billing@contoso.com" in next(f["detail"] for f in r["failures"] if f["check"] == "placeholders")
