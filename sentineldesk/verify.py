"""⑤ VERIFY: deterministic rules that check the LLM's draft before anyone sees it.

    from sentineldesk.verify import verify
    result = verify(draft, retrieved, ticket_text, lang)
    if not result["passed"]:
        ...  # escalate: never show the draft; result["failures"] lists every reason

Inputs
    draft        {"reply": str, "cited_ids": [str], "unsupported_asks": [str]}  (the LLM's JSON)
    retrieved    search() output: [{"ticket_id", "answer", ...}, ...]
    ticket_text  the customer's ticket, already PII-masked (subject + "\\n" + body)
    lang         "en" or "de", from sentineldesk.language

Checks (all run every time, so the agent sees every failure)
    format            reply is a non-empty string within MAX_WORDS; list fields are lists        VERIFY_FAILED
    citations         cited ids (field + inline TKT-xxxxx) are all retrieved, and at least one  VERIFY_FAILED
    numbers           every number / amount / % / duration / date in the reply appears in a     VERIFY_FAILED
                      retrieved answer or in the customer's ticket (team decision)
    placeholders      <placeholders> come from the sources; no invented email/phone/account     VERIFY_FAILED
    language          the reply is in the ticket's language (enforced from MIN_WORDS_FOR_LANG)  VERIFY_FAILED
    unsupported_asks  the LLM listed nothing it could not ground                                UNSUPPORTED_ASK

Pure Python: only `re`, the repo's pii.mask() and sentineldesk.language. Runs in about a millisecond.
"""
from __future__ import annotations

import re

from pii import mask
from sentineldesk.language import detect_language_with_confidence

MAX_WORDS = 250
MIN_WORDS_FOR_LANG = 8  # the language detector is unreliable on very short replies

VERIFY_FAILED = "VERIFY_FAILED"
UNSUPPORTED_ASK = "UNSUPPORTED_ASK"

_TKT = re.compile(r"\bTKT-\d{5}\b")
_PLACEHOLDER = re.compile(r"<[a-z_]+>")

# ---------------------------------------------------------------- number extraction
_NUM = r"\d+(?:[.,]\d+)*"
_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
    "ein": 1, "eins": 1, "einen": 1, "einem": 1, "einer": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12,
}
_UNITS = {  # duration words (en + de) -> one canonical unit
    "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute", "minuten": "minute",
    "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour", "stunde": "hour", "stunden": "hour",
    "day": "day", "days": "day", "business day": "day", "business days": "day", "working day": "day",
    "working days": "day", "tag": "day", "tage": "day", "tagen": "day", "werktag": "day", "werktage": "day",
    "werktagen": "day", "arbeitstag": "day", "arbeitstage": "day", "arbeitstagen": "day",
    "week": "week", "weeks": "week", "woche": "week", "wochen": "week",
    "month": "month", "months": "month", "monat": "month", "monate": "month", "monaten": "month",
}
_UNIT_RE = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
_CUR = {"€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "$": "USD", "usd": "USD", "dollar": "USD",
        "dollars": "USD", "£": "GBP", "gbp": "GBP", "chf": "CHF"}
_CUR_RE = r"€|\$|£|eur|euros?|usd|dollars?|gbp|chf"

_PATTERNS = [  # most specific first; each match is blanked out so later patterns can't re-read it
    ("date", re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("time", re.compile(r"\b\d{1,2}:\d{2}\b")),
    ("always", re.compile(r"\b24\s?/\s?7\b")),
    ("cur", re.compile(rf"(?P<c1>{_CUR_RE})\s?(?P<n1>{_NUM})|(?P<n2>{_NUM})\s?(?P<c2>{_CUR_RE})(?![a-z])", re.I)),
    ("pct", re.compile(rf"(?P<n>{_NUM})\s?(?:%|percent|per cent|prozent)", re.I)),
    ("dur", re.compile(rf"(?P<n>{_NUM})[\s-]?(?P<u>{_UNIT_RE})\b", re.I)),
    ("num", re.compile(_NUM)),
]


def _readings(raw: str) -> set[str]:
    """Every value a number string could mean. '1.000' -> {'1000', '1'}; '1.000,50' -> {'1000.5'}; '8.0.1' kept raw."""
    out = {raw}
    seps = re.findall(r"[.,]", raw)
    groups = re.split(r"[.,]", raw)
    if not seps:
        return {str(int(raw))}
    if all(len(g) == 3 for g in groups[1:]):  # thousands separators only, e.g. 1.000 / 1,000,000
        out.add(str(int("".join(groups))))
    head, tail = groups[:-1], groups[-1]  # last separator as the decimal point
    if all(len(g) == 3 for g in head[1:]):
        val = float(f"{''.join(head)}.{tail}")
        out.add(f"{val:g}" if val != int(val) else str(int(val)))
    return out


def extract_values(text: str) -> list[tuple[str, frozenset, str]]:
    """Find every number-bearing claim. Returns (kind, possible normalised values, the text as written)."""
    text = _PLACEHOLDER.sub(" ", _TKT.sub(" ", text or ""))
    # number words count only before a unit or currency ("three days", "drei Tage"), never alone ("one of us")
    text = re.sub(rf"\b({'|'.join(_WORD_NUMS)})\b(?=[\s-]+(?:{_UNIT_RE}|{_CUR_RE})\b)",
                  lambda m: str(_WORD_NUMS[m.group(1).lower()]), text, flags=re.I)
    found = []
    for kind, rx in _PATTERNS:
        def take(m, kind=kind):
            s = m.group(0)
            if kind in ("date", "time"):
                vals = {re.sub(r"\s", "", s)}
            elif kind == "always":
                vals = {"24/7"}
            elif kind == "cur":
                n, c = (m.group("n1"), m.group("c1")) if m.group("n1") else (m.group("n2"), m.group("c2"))
                vals = {f"{_CUR[c.lower()]}:{r}" for r in _readings(n)}
            elif kind == "pct":
                vals = {f"%:{r}" for r in _readings(m.group("n"))}
            elif kind == "dur":
                unit = _UNITS[re.sub(r"\s+", " ", m.group("u").lower())]
                vals = {f"{unit}:{r}" for r in _readings(m.group("n"))}
            else:
                vals = _readings(s)
            found.append((kind, frozenset(vals), s.strip()))
            return " " * len(s)
        text = rx.sub(take, text)
    return found


def _source_values(texts: list[str]) -> set[str]:
    """All values in the sources. A typed value (EUR:50) also makes its bare number (50) available."""
    vals: set[str] = set()
    for t in texts:
        for _, vs, _ in extract_values(t):
            vals |= vs
            vals |= {v.split(":", 1)[1] for v in vs if ":" in v and v not in ("24/7",) and not re.match(r"\d", v)}
    return vals


# ---------------------------------------------------------------- the checks
def _check_format(draft) -> tuple[bool, str, str, list, list]:
    if not isinstance(draft, dict):
        return False, f"draft is {type(draft).__name__}, not a dict", "", [], []
    reply, cited, asks = draft.get("reply"), draft.get("cited_ids"), draft.get("unsupported_asks")
    problems = []
    if not isinstance(reply, str) or not reply.strip():
        problems.append("reply is missing or empty")
        reply = reply if isinstance(reply, str) else ""
    elif len(reply.split()) > MAX_WORDS:
        problems.append(f"reply has {len(reply.split())} words (max {MAX_WORDS})")
    if not isinstance(cited, list):
        problems.append("cited_ids is not a list")
        cited = []
    if not isinstance(asks, list):
        problems.append("unsupported_asks is not a list")
        asks = []
    ok = not problems
    return ok, "; ".join(problems) or f"{len(reply.split())} words, fields well-formed", reply, cited, asks


def _check_citations(reply: str, cited: list, retrieved_ids: set[str]) -> tuple[bool, str]:
    ids = {str(c).strip() for c in cited} | set(_TKT.findall(reply))
    bad = sorted(ids - retrieved_ids)
    good = sorted(ids & retrieved_ids)
    if bad:
        return False, f"cites {bad}, not among the retrieved tickets"
    if not good:
        return False, "no ticket is cited"
    return True, f"cites {good}, all retrieved"


def _check_numbers(reply: str, source_texts: list[str]) -> tuple[bool, str]:
    claims = extract_values(reply)
    if not claims:
        return True, "no numbers in the reply"
    allowed = _source_values(source_texts)
    missing = [s for _, vs, s in claims if not (vs & allowed)]
    if missing:
        return False, f"not found in any retrieved answer or the ticket: {missing}"
    return True, f"all {len(claims)} grounded: {[s for _, _, s in claims][:6]}"


def _check_placeholders(reply: str, source_texts: list[str]) -> tuple[bool, str]:
    allowed = set().union(*(_PLACEHOLDER.findall(t) for t in source_texts)) if source_texts else set()
    unknown = sorted(set(_PLACEHOLDER.findall(reply)) - allowed)
    leaked = sorted(set(_PLACEHOLDER.findall(mask(reply))) - set(_PLACEHOLDER.findall(reply)))
    problems = []
    if unknown:
        problems.append(f"placeholders not in any source: {unknown}")
    if leaked:
        problems.append(f"reply contains personal data that would be masked as {leaked} (invented by the model)")
    if problems:
        return False, "; ".join(problems)
    return True, f"placeholders kept: {sorted(set(_PLACEHOLDER.findall(reply))) or 'none used'}"


def _check_language(reply: str, lang: str) -> tuple[bool, str]:
    clean = _PLACEHOLDER.sub(" ", _TKT.sub(" ", reply))
    if len(clean.split()) < MIN_WORDS_FOR_LANG:
        return True, f"too short to judge ({len(clean.split())} words); not enforced"
    pred = detect_language_with_confidence(body=clean)
    if pred.language != lang:
        return False, f"reply detected as '{pred.language}' (confidence {pred.confidence:.2f}), ticket is '{lang}'"
    return True, f"reply is '{pred.language}' (confidence {pred.confidence:.2f}), matches the ticket"


def _check_unsupported(asks: list) -> tuple[bool, str]:
    asks = [str(a) for a in asks if str(a).strip()]
    if asks:
        return False, f"model could not ground: {asks[:3]}"
    return True, "none"


# ---------------------------------------------------------------- public entry point
def verify(draft: dict, retrieved: list[dict], ticket_text: str, lang: str) -> dict:
    """Run every check. passed is True only if all pass; failures carry the code the gate escalates with."""
    retrieved = retrieved or []
    retrieved_ids = {str(r.get("ticket_id")) for r in retrieved}
    answers = [str(r.get("answer") or "") for r in retrieved]
    sources = answers + [ticket_text or ""]  # team decision: the customer's own numbers are allowed

    fmt_ok, fmt_ev, reply, cited, asks = _check_format(draft)
    results = [
        ("format", VERIFY_FAILED, fmt_ok, fmt_ev),
        ("citations", VERIFY_FAILED, *_check_citations(reply, cited, retrieved_ids)),
        ("numbers", VERIFY_FAILED, *_check_numbers(reply, sources)),
        ("placeholders", VERIFY_FAILED, *_check_placeholders(reply, sources)),
        ("language", VERIFY_FAILED, *_check_language(reply, lang)),
        ("unsupported_asks", UNSUPPORTED_ASK, *_check_unsupported(asks)),
    ]
    checks = [{"check": n, "passed": ok, "evidence": ev} for n, _, ok, ev in results]
    failures = [{"check": n, "code": code, "detail": ev} for n, code, ok, ev in results if not ok]
    return {"passed": not failures, "failures": failures, "checks": checks}
