"""⑤ VERIFY: deterministic rules that check the LLM's draft before anyone sees it.

    from sentineldesk.verify import verify
    result = verify(draft, retrieved, ticket_text, lang)
    if not result["passed"]:
        ...  # escalate: never show the draft; result["failures"] lists every reason

Inputs
    draft        {"reply": str, "cited_ids": [str], "unsupported_asks": [str]}  (the LLM's JSON)
    retrieved    search() output: [{"ticket_id", "answer", "language", ...}, ...]
    ticket_text  the customer's ticket, already PII-masked (subject + "\\n" + body)
    lang         "en" or "de", from sentineldesk.language

Checks (all run every time, so the agent sees every failure)
    format            reply non-empty, <= MAX_WORDS words and MAX_CHARS characters; list fields   VERIFY_FAILED
    citations         cited ids (field + inline TKT-xxxxx) all retrieved, at least one            VERIFY_FAILED
    numbers           amounts / % / quantities with a unit must appear, same unit, in a retrieved  VERIFY_FAILED
                      answer; plain numbers, dates and times in a retrieved answer or the ticket
    placeholders      <placeholders> come from the sources; no invented email / phone / account /  VERIFY_FAILED
                      link (text quoted verbatim from a source is not counted as invented)
    language          the whole reply and every full sentence are in the ticket's language        VERIFY_FAILED
    unsupported_asks  the LLM listed nothing it could not ground                                  UNSUPPORTED_ASK

Pure Python: re, difflib, decimal, the repo's pii module and sentineldesk.language. About a millisecond per draft.
"""
from __future__ import annotations

import difflib
import re
from decimal import Decimal, InvalidOperation

from pii import mask, normalize_placeholders
from sentineldesk.language import detect_language_with_confidence

MAX_WORDS = 250
MAX_CHARS = 3000          # one 40k-character "word" must not slip past MAX_WORDS or slow the checks down
MIN_WORDS_FOR_LANG = 8    # alphabetic words; the detector is unreliable below this
MIN_WORDS_PER_SENTENCE = 6
SENTENCE_LANG_CONF = 0.7  # a full sentence in the other language at this confidence fails the reply
MAX_DIGITS = 30           # longer numbers are not read (and so never match): no overflow, no int-size limits

VERIFY_FAILED = "VERIFY_FAILED"
UNSUPPORTED_ASK = "UNSUPPORTED_ASK"

_TKT = re.compile(r"\bTKT-\d{5}\b")
_PH_ANY = re.compile(r"<[A-Za-z][A-Za-z_ -]{0,30}>")   # any <placeholder>, any case
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\]\)]+", re.I)
_OBF_EMAIL = re.compile(r"\b[\w.+-]+\s*[(\[]\s*(?:at|ät)\s*[)\]]\s*[\w-]+(?:\s*[(\[]\s*(?:dot|punkt)\s*[)\]]\s*\w+)+", re.I)
_ALPHA_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]{2,}")

# ---------------------------------------------------------------- units (en + de -> one canonical name)
_UNIT_WORDS = {
    "second": "second seconds sec secs sekunde sekunden",
    "minute": "minute minutes min mins minute minuten",
    "hour": "hour hours hr hrs h std stunde stunden",
    "day": "day days tag tage tagen tages werktag werktage werktagen arbeitstag arbeitstage arbeitstagen",
    "week": "week weeks woche wochen",
    "month": "month months monat monate monaten monats",
    "year": "year years yr yrs jahr jahre jahren jahres",
    "times": "times time mal fach",
    "data_kb": "kb", "data_mb": "mb", "data_gb": "gb", "data_tb": "tb",
    "user": "user users nutzer benutzer benutzerin",
    "seat": "seat seats platz plätze",
    "license": "license licenses licence licences lizenz lizenzen",
    "device": "device devices gerät geräte",
}
_UNITS = {w: canon for canon, words in _UNIT_WORDS.items() for w in words.split()}
_FILLERS = ("more additional extra full further business working calendar weitere weiteren zusätzliche "
            "zusätzlichen volle vollen ganze ganzen").split()
_CURRENCY = {"€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "$": "USD", "usd": "USD", "dollar": "USD",
             "dollars": "USD", "£": "GBP", "gbp": "GBP", "chf": "CHF", "cent": "CENT", "cents": "CENT"}
_PCT_WORDS = ("%", "percent", "per cent", "prozent", "pct")


def _alt(words) -> str:
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


_UNIT_RE, _FILL_RE = _alt(_UNITS), _alt(_FILLERS)
_CUR_SYM_RE, _CUR_WORD_RE = r"€|\$|£", _alt(k for k in _CURRENCY if k.isalpha())
_PCT_RE = _alt(_PCT_WORDS)
_GAP = rf"(?:[\s-]+(?:{_FILL_RE}))*[\s-]?"   # optional filler words between a number and its unit
_NUM = r"\d{1,3}(?:[ \u00a0\u202f'\u2019]\d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,]\d+)*"  # 1 000,50 / 1'000 / 1.000,50

# ---------------------------------------------------------------- number words (en + de)
_EN_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_EN_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_EN_SCALE = {"hundred": 100, "thousand": 1000, "dozen": 12}
_EN_WORDS = set(_EN_SMALL) | set(_EN_TENS) | set(_EN_SCALE)
_DE_SMALL = {"null": 0, "ein": 1, "eins": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1, "zwei": 2, "drei": 3,
             "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12,
             "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18,
             "neunzehn": 19}
_DE_TENS = {"zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
            "siebzig": 70, "achtzig": 80, "neunzig": 90}
_DE_STEM = re.compile(r"[a-zäöüß]*(?:ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf|zig|ßig|ssig|"
                      r"hundert|tausend|null)[a-zäöüß]*")


def _en_value(phrase: str) -> int | None:
    toks = [t for t in re.split(r"[\s-]+", phrase.lower()) if t and t != "and"]
    if toks and toks[0] in ("a", "an"):
        toks = toks[1:]
        if not toks or toks[0] not in _EN_SCALE:
            return None
        toks = ["one"] + toks
    total = cur = 0
    for t in toks:
        if t in _EN_SMALL:
            cur += _EN_SMALL[t]
        elif t in _EN_TENS:
            cur += _EN_TENS[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t == "thousand":
            total, cur = total + (cur or 1) * 1000, 0
        elif t == "dozen":
            cur = (cur or 1) * 12
        else:
            return None
    return total + cur


def _de_value(word: str) -> int | None:
    w = word.lower()
    if w in _DE_SMALL:
        return _DE_SMALL[w]
    if w in _DE_TENS:
        return _DE_TENS[w]
    for scale, mult in (("tausend", 1000), ("hundert", 100)):
        if scale in w:
            head, tail = w.split(scale, 1)
            h = _de_value(head) if head else 1
            t = _de_value(tail.removeprefix("und")) if tail else 0
            return None if h is None or t is None else h * mult + t
    if "und" in w:
        ones, tens = w.split("und", 1)
        if ones in _DE_SMALL and 1 <= _DE_SMALL[ones] <= 9 and tens in _DE_TENS:
            return _DE_SMALL[ones] + _DE_TENS[tens]
    return None


_EN_RUN = re.compile(rf"\b(?:(?:a|an)[\s-]+)?(?:{_alt(_EN_WORDS)})(?:[\s-]+(?:and[\s-]+)?(?:{_alt(_EN_WORDS)}))*\b", re.I)
_FOLLOWED_BY_UNIT = re.compile(rf"{_GAP}(?:{_UNIT_RE}|{_CUR_WORD_RE}|{_PCT_RE})\b|\s?(?:{_CUR_SYM_RE}|%)", re.I)


def _convert_number_words(text: str, unreadable: list[str]) -> str:
    """Number words before a unit/currency/percent become digits ("twenty-four hours" -> "24 hours").
    Number words that are not quantities ("one of our agents") are left alone. A quantity written in words
    that can't be read is recorded in `unreadable`, so the numbers check fails instead of skipping it."""
    def handle(span: str, value: int | None) -> str:
        if value is None:
            unreadable.append(span)
            return " " * len(span)
        return str(value)

    def en(m):
        if not _FOLLOWED_BY_UNIT.match(text, m.end()):
            return m.group(0)
        return handle(m.group(0), _en_value(m.group(0)))

    text = _EN_RUN.sub(en, text)

    def de(m):
        if not _FOLLOWED_BY_UNIT.match(text, m.end()):
            return m.group(0)
        word = m.group(0)
        return handle(word, _de_value(word))

    return re.sub(rf"\b(?:{_DE_STEM.pattern})\b", de, text, flags=re.I)


# ---------------------------------------------------------------- reading a number string
def _canon(d: Decimal) -> str:
    return format(d.normalize(), "f")


def _readings(raw: str, lang: str | None, money: bool = False) -> frozenset[str]:
    """What a number string means, read the way `lang` writes numbers.
    en: ',' groups thousands, '.' is the decimal point. de: the reverse. Both present: the last one is decimal.
    A single separator followed by exactly 3 digits is the only ambiguous case; it is read by `lang` (both
    readings only if the language is unknown). Version-like strings (8.0.1) are kept as written."""
    raw = re.sub(r"[ \u00a0\u202f'\u2019]", "", raw)  # space/apostrophe thousands separators
    if sum(c.isdigit() for c in raw) > MAX_DIGITS:
        return frozenset({f"raw:{raw}"})
    seps = set(re.findall(r"[.,]", raw))
    groups = re.split(r"[.,]", raw)
    try:
        if not seps:
            return frozenset({_canon(Decimal(raw))})
        if len(seps) == 2:  # both: the last separator is the decimal point
            dec = raw[max(raw.rfind("."), raw.rfind(","))]
            whole, frac = raw.rsplit(dec, 1)
            return frozenset({_canon(Decimal(re.sub(r"[.,]", "", whole) + "." + frac))})
        sep = seps.pop()
        if len(groups) > 2:  # 1.000.000 / 1,000,000, or a version string
            if all(len(g) == 3 for g in groups[1:]):
                return frozenset({_canon(Decimal("".join(groups)))})
            return frozenset({f"raw:{raw}"})
        whole, frac = groups
        thousands = _canon(Decimal(whole + frac))
        decimal = _canon(Decimal(f"{whole}.{frac}"))
        if len(frac) != 3:
            return frozenset({decimal})
        if money:  # no currency has 3 decimals: "€1.000" and "€2,500" are thousands in any language
            return frozenset({thousands})
        if lang == "en":
            return frozenset({thousands if sep == "," else decimal})
        if lang == "de":
            return frozenset({thousands if sep == "." else decimal})
        return frozenset({thousands, decimal})
    except (InvalidOperation, ValueError):
        return frozenset({f"raw:{raw}"})


# ---------------------------------------------------------------- dates and times
def _iso(y: int, m: int, d: int) -> str | None:
    if y < 100:
        y += 2000
    return f"{y:04d}-{m:02d}-{d:02d}" if 1 <= m <= 12 and 1 <= d <= 31 else None


def _date_readings(s: str) -> frozenset[str]:
    s = s.strip()
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        iso = {_iso(int(m[1]), int(m[2]), int(m[3]))}
    elif m := re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", s):
        iso = {_iso(int(m[3]), int(m[2]), int(m[1]))}
    elif m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s):  # en may be MM/DD, others DD/MM
        iso = {_iso(int(m[3]), int(m[2]), int(m[1])), _iso(int(m[3]), int(m[1]), int(m[2]))}
    else:
        iso = set()
    iso.discard(None)
    return frozenset({f"date:{x}" for x in iso} or {f"raw:{s}"})


# ---------------------------------------------------------------- extraction
_PATTERNS = [  # most specific first; each match is blanked so later patterns can't re-read it
    ("date", re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("time", re.compile(r"\b(\d{1,2}):(\d{2})\b")),
    ("always", re.compile(r"\b24\s?/\s?7\b")),
    ("cur", re.compile(rf"(?P<c1>{_CUR_SYM_RE}|\b(?:{_CUR_WORD_RE})\b)\s?(?P<n1>{_NUM})"
                       rf"|(?P<n2>{_NUM}){_GAP}(?P<c2>{_CUR_SYM_RE}|\b(?:{_CUR_WORD_RE})\b)", re.I)),
    ("pct", re.compile(rf"(?P<n>{_NUM}){_GAP}(?:{_PCT_RE})(?![a-z])", re.I)),
    ("qty", re.compile(rf"(?P<n>{_NUM}){_GAP}(?P<u>{_UNIT_RE})\b", re.I)),
    ("num", re.compile(_NUM)),
]
TYPED = ("cur", "pct", "qty")


def extract_values(text: str, lang: str | None = None) -> list[tuple[str, frozenset, str]]:
    """Every number-bearing claim as (kind, possible normalised values, the text as written).
    A quantity written in words that can't be read comes back with no values, so it never matches."""
    text = _PH_ANY.sub(" ", _TKT.sub(" ", text or ""))
    unreadable: list[str] = []
    text = _convert_number_words(text, unreadable)
    found = [("unreadable", frozenset(), s.strip()) for s in unreadable]
    for kind, rx in _PATTERNS:
        def take(m, kind=kind):
            s = m.group(0)
            if kind == "date":
                vals = _date_readings(s)
            elif kind == "time":
                vals = frozenset({f"time:{int(m.group(1))}:{m.group(2)}"})
            elif kind == "always":
                vals = frozenset({"always:24/7"})
            elif kind == "cur":
                n, c = (m.group("n1"), m.group("c1")) if m.group("n1") else (m.group("n2"), m.group("c2"))
                vals = frozenset(f"{_CURRENCY[c.lower()]}:{r}" for r in _readings(n, lang, money=True))
            elif kind == "pct":
                vals = frozenset(f"%:{r}" for r in _readings(m.group("n"), lang))
            elif kind == "qty":
                vals = frozenset(f"{_UNITS[m.group('u').lower()]}:{r}" for r in _readings(m.group("n"), lang))
            else:
                vals = _readings(s, lang)
            found.append((kind, vals, s.strip()))
            return " " * len(s)
        text = rx.sub(take, text)
    return found


# ---------------------------------------------------------------- the checks
def _check_format(draft):
    if not isinstance(draft, dict):
        return False, f"draft is {type(draft).__name__}, not a dict", "", [], []
    reply, cited, asks = draft.get("reply"), draft.get("cited_ids"), draft.get("unsupported_asks")
    problems = []
    if not isinstance(reply, str) or not reply.strip():
        problems.append("reply is missing or empty")
        reply = reply if isinstance(reply, str) else ""
    else:
        if len(reply.split()) > MAX_WORDS:
            problems.append(f"reply has {len(reply.split())} words (max {MAX_WORDS})")
        if len(reply) > MAX_CHARS:
            problems.append(f"reply has {len(reply)} characters (max {MAX_CHARS})")
    if not isinstance(cited, list):
        problems.append("cited_ids is not a list")
        cited = []
    if not isinstance(asks, list):
        problems.append("unsupported_asks is not a list")
        asks = []
    ev = "; ".join(problems) or f"{len(reply.split())} words, fields well-formed"
    return not problems, ev, reply[:MAX_CHARS], cited, asks  # later checks never see more than MAX_CHARS


def _check_citations(reply, cited, retrieved_ids):
    ids = {c.strip() for c in cited if isinstance(c, str)} | set(_TKT.findall(reply))
    if any(not isinstance(c, str) for c in cited):
        return False, f"cited_ids contains non-text entries: {[c for c in cited if not isinstance(c, str)][:3]}"
    bad, good = sorted(ids - retrieved_ids), sorted(ids & retrieved_ids)
    if bad:
        return False, f"cites {bad}, not among the retrieved tickets"
    if not good:
        return False, "no ticket is cited"
    return True, f"cites {good}, all retrieved"


def _check_numbers(reply, lang, answers, ticket_text):
    """Amounts, percentages and quantities with a unit must be in a retrieved answer with the same unit.
    Plain numbers, dates and times may also come from the customer's ticket (versions, order numbers)."""
    claims = extract_values(reply, lang)
    if not claims:
        return True, "no numbers in the reply"
    typed_ok, plain_ok = set(), set()
    for text, src_lang in answers:
        for kind, vals, _ in extract_values(text, src_lang):
            (typed_ok if kind in TYPED else plain_ok).update(vals)
    for kind, vals, _ in extract_values(ticket_text, lang):
        if kind not in TYPED:
            plain_ok.update(vals)
    missing = [s for kind, vals, s in claims if not (vals & (typed_ok if kind in TYPED else plain_ok))]
    if missing:
        return False, f"not grounded (amounts, % and durations need a retrieved answer): {missing}"
    return True, f"all {len(claims)} grounded: {[s for _, _, s in claims][:6]}"


def _check_placeholders(reply, sources):
    allowed = {p.lower() for s in sources for p in _PH_ANY.findall(s)}
    unknown = sorted({p for p in _PH_ANY.findall(reply) if p.lower() not in allowed})
    problems = []
    if unknown:
        problems.append(f"placeholders not in any source: {unknown}")

    # Personal data the model wrote itself: mask the reply (existing placeholders removed first, so a new
    # <email> can't hide behind an old one) and list what got replaced, unless it is quoted from a source.
    plain = _PH_ANY.sub(" ", reply)
    masked = mask(plain)
    squash = lambda s: re.sub(r"\s+", " ", s).strip()
    corpus = squash(" ".join(sources))
    leaked = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, plain, masked, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        while i1 > 0 and not plain[i1 - 1].isspace():  # widen to whole words for the check and the evidence
            i1 -= 1
        while i2 < len(plain) and not plain[i2].isspace():
            i2 += 1
        span = squash(plain[i1:i2]).strip(".,;:!?()")
        tags = sorted(set(_PH_ANY.findall(masked[max(0, j1 - 40):j2 + 40]))) or ["<pii>"]
        if span and span not in corpus and not any(span in x for x in leaked):
            leaked.append(f"{span!r} -> {'/'.join(tags)}")
    leaked += [f"{u!r} (link)" for u in _URL.findall(reply) if u.rstrip(".,;") not in corpus]
    leaked += [f"{e!r} (email)" for e in _OBF_EMAIL.findall(reply)]
    if leaked:
        problems.append(f"personal data or links not in any source (invented by the model): {leaked[:4]}")
    if problems:
        return False, "; ".join(problems)
    return True, f"placeholders kept: {sorted(set(_PH_ANY.findall(reply))) or 'none used'}"


def _check_language(reply, lang):
    if lang not in ("en", "de"):
        return False, f"ticket language {lang!r} is not en/de"
    clean = _URL.sub(" ", _PH_ANY.sub(" ", _TKT.sub(" ", reply)))
    n_words = len(_ALPHA_WORD.findall(clean))
    if n_words < MIN_WORDS_FOR_LANG:
        return True, f"too short to judge ({n_words} words); not enforced"
    whole = detect_language_with_confidence(body=clean)
    if whole.language != lang:
        return False, f"reply detected as '{whole.language}' ({whole.confidence:.2f}), ticket is '{lang}'"
    off = []
    for sent in re.split(r"[.!?\n;:]+", clean):
        if len(_ALPHA_WORD.findall(sent)) >= MIN_WORDS_PER_SENTENCE:
            p = detect_language_with_confidence(body=sent)
            if p.language != lang and p.confidence >= SENTENCE_LANG_CONF:
                off.append(sent.strip()[:60])
    if off:
        return False, f"sentences not in '{lang}': {off[:2]}"
    return True, f"reply and all full sentences are '{lang}' ({whole.confidence:.2f})"


def _check_unsupported(asks):
    asks = [str(a) for a in asks if str(a).strip()]
    if asks:
        return False, f"model could not ground: {asks[:3]}"
    return True, "none"


# ---------------------------------------------------------------- public entry point
def verify(draft: dict, retrieved: list[dict], ticket_text: str, lang: str) -> dict:
    """Run every check. passed is True only if all pass; failures carry the code the gate escalates with."""
    rows = [r for r in (retrieved if isinstance(retrieved, list) else [])
            if isinstance(r, dict) and isinstance(r.get("ticket_id"), str) and r["ticket_id"].strip()]
    retrieved_ids = {r["ticket_id"].strip() for r in rows}
    lang = str(lang or "").strip().lower()
    ticket_text = "" if ticket_text is None else str(ticket_text)
    answers = []
    for r in rows:
        text = normalize_placeholders(str(r.get("answer") or ""))
        src_lang = str(r.get("language") or "").lower()
        if src_lang not in ("en", "de"):
            src_lang = detect_language_with_confidence(body=text).language if text.strip() else None
        answers.append((text, src_lang))
    sources = [a for a, _ in answers] + [ticket_text]

    fmt_ok, fmt_ev, reply, cited, asks = _check_format(draft)
    results = [
        ("format", VERIFY_FAILED, fmt_ok, fmt_ev),
        ("citations", VERIFY_FAILED, *_check_citations(reply, cited, retrieved_ids)),
        ("numbers", VERIFY_FAILED, *_check_numbers(reply, lang if lang in ("en", "de") else None, answers, ticket_text)),
        ("placeholders", VERIFY_FAILED, *_check_placeholders(reply, sources)),
        ("language", VERIFY_FAILED, *_check_language(reply, lang)),
        ("unsupported_asks", UNSUPPORTED_ASK, *_check_unsupported(asks)),
    ]
    checks = [{"check": n, "passed": ok, "evidence": ev} for n, _, ok, ev in results]
    failures = [{"check": n, "code": code, "detail": ev} for n, code, ok, ev in results if not ok]
    return {"passed": not failures, "failures": failures, "checks": checks}
