# Language Detection Guide (`sentineldesk.language`)

## 1. What does this module do?

When a customer sends a support ticket (`subject` + `body`), the very first step in SentinelDesk is figuring out **what language they are writing in** (by default **English `en`** or **German `de`**, with built-in support for adding more languages at any time).

### Why don't we just read the dataset's `language` column?
1. **Hackathon Rule**: The problem statement in [README.md](file:///Users/hung/Desktop/git/HackathonGDG/README.md#L127) explicitly states: *"detect it from the text at demo time. Don't read the `language` column."*
2. **Mislabeled Data in the Raw Dataset**: When we inspected all 61,765 tickets in `Tobi-Bueck/customer-support-tickets`, we found that **8.7% (5,377 tickets) have the wrong label** in the raw `language` column:
   - Example `#32`: labeled `de` in the CSV, but the text is pure English (`"Dear Customer Support Team, I am reaching out to report a persistent problem..."`).
   - Example `#12269`: labeled `en` in the CSV, but the text is pure German (`"Probleme mit dem Smart-Türklingel — Ich beschreibe das Problem..."`).
3. **Crash-Proof on Empty Inputs**: 5,299 tickets have an empty `subject` (`None` or `NaN`), and 2 tickets have an empty `body`. This module handles missing or empty inputs safely and never crashes.

---

## 2. Quick Start: How to use it in your code

Teammates working on **Triage**, **Retrieval**, **UI**, or **LLM Drafting** only need to import two functions:

### A. Get the language code (`'en'` or `'de'`)
```python
from sentineldesk import detect_language

lang = detect_language(
    subject="Wesentlicher Sicherheitsvorfall",
    body="Sehr geehrtes Team, der Server ist ausgefallen.",
)
print(lang)  # -> 'de'
```

### B. Get language + confidence score (for the Live Console UI)
```python
from sentineldesk import detect_language_with_confidence

result = detect_language_with_confidence(
    subject="VPN is down",
    body="Our hospital staff cannot access patient records.",
)
print(result.language)    # -> 'en'
print(result.confidence)  # -> 0.95 (ranges from 0.50 to 1.00)
print(result.scores)      # -> {'en': 10, 'de': 0}
```

### C. Process a batch of tickets at once
```python
from sentineldesk import detect_language_batch

langs = detect_language_batch(df["subject"], df["body"])
```

---

## 3. How does it work under the hood?

Instead of calling a slow LLM or heavy neural network, it uses a **3-Step Evidence Score** that runs in **0.02 milliseconds (20 microseconds)** per ticket with zero external dependencies:

```text
Incoming Ticket (subject + body)
       │
       ├─► Step 1: Common Words Check (+2 points per match)
       │     • English words: "the", "and", "please", "dear", "issue", "unable"...
       │     • German words : "ich", "nicht", "bitte", "sehr", "fehler", "zugriff"...
       │
       ├─► Step 2: Special Letters Check (+3 points per character)
       │     • German letters: ä, ö, ü, ß
       │
       └─► Step 3: Word Ending Tie-Breaker (+1 point, only used when scores are close)
             • English endings: -tion, -ment, -ing, -able (e.g. "Configuration failed")
             • German endings : -ung, -keit, -heit, -ieren (e.g. "Systemaktualisierung")
```

Whichever language gets the higher total score wins. The larger the gap between the top score and the runner-up, the higher the `confidence` score.

---

## 4. Extensibility: How to add more languages in the future

We built this module around **Language Profiles (`LanguageProfile`)** so nothing is hardcoded to just English and German. You can add new languages without touching the core scoring engine.

### Option 1: Enable built-in French (`fr`) and Spanish (`es`)
French and Spanish profiles are already included out-of-the-box:

```python
from sentineldesk import (
    DE_PROFILE,
    EN_PROFILE,
    ES_PROFILE,
    FR_PROFILE,
    LanguageDetector,
)

# Create a detector that supports 4 languages (en, de, fr, es)
detector = LanguageDetector(
    profiles=[EN_PROFILE, DE_PROFILE, FR_PROFILE, ES_PROFILE]
)

print(
    detector.detect(
        "Problème de compte", "Bonjour, merci pour votre aide avec notre compte."
    )
)
# -> 'fr'
```

### Option 2: Register any brand-new language at runtime (e.g., Italian `it`)
Create a `LanguageProfile` with common words and special characters, then call `register_profile()`:

```python
import re
from sentineldesk import DE_PROFILE, EN_PROFILE, LanguageDetector, LanguageProfile

detector = LanguageDetector(profiles=[EN_PROFILE, DE_PROFILE])

italian = LanguageProfile(
    code="it",
    name="Italian",
    distinctive_chars=frozenset("àèéìòù"),
    stopwords=frozenset({
        "il", "lo", "gli", "della", "nella", "sono", "siamo", "abbiamo",
        "buongiorno", "grazie", "problema", "nostro", "sistema", "assistenza",
    }),
    suffix_pattern=re.compile(r"(?:zione|zioni|mento|menti)$"),
)

detector.register_profile(italian)

print(
    detector.detect(
        "Problema con il sistema", "Buongiorno, grazie per l'assistenza."
    )
)
# -> 'it'
```

---

## 5. Speed & Accuracy Results

You can run the unit tests and full-dataset benchmark at any time:

```bash
# 1. Run 23 unit & extensibility tests (takes 0.06 seconds)
PYTHONPATH=. uv run --with pytest pytest -v tests/test_language.py

# 2. Run benchmark on all 61,765 real tickets from Hugging Face
PYTHONPATH=. uv run --with datasets --with pandas --with langdetect python3 scripts/verify_language_detector.py
```

| Test Metric | Result |
| :--- | :--- |
| **Total tickets tested** | **61,765 tickets** (finished in **1.28 seconds** total) |
| **Average latency per ticket** | **0.021 ms** (~21 microseconds; **48,000+ tickets/sec**) |
| **Crashes on empty `subject` / `body`** | **0 crashes** (verified across 5,299 missing subjects and 2 missing bodies) |
| **Agreement with `langdetect` (N=1,000 & N=2,000)** | **100.00% agreement** (while running **67× faster**) |
