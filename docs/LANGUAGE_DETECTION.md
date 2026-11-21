# SentinelDesk Language Detection (`sentineldesk.language`)

## 1. Purpose & Specification Compliance
In accordance with the [README.md](file:///Users/hung/Desktop/git/HackathonGDG/README.md) rules:
- **Runtime text-only inference**: Language is inferred strictly from incoming ticket text (`subject` and `body`) at runtime.
- **Zero leakage / No `language` column dependency**: The detector never reads the dataset's `language` column. (Auditing the raw dataset revealed that **~8.7% / 5,377 rows** in `Tobi-Bueck/customer-support-tickets` have an inverted or mislabeled `language` column, such as pure English tickets labeled `de` and pure German tickets labeled `en`).
- **Crash-proof on missing inputs**: Safely handles `None`, `float("nan")`, empty strings, whitespace-only strings, and numeric/symbol-only inputs (`subject` is missing in 5,299 rows and `body` in 2 rows of the dataset).

---

## 2. Scoring Algorithm & Architecture
`LanguageDetector` uses a deterministic, 3-tier evidence scoring pipeline per active `LanguageProfile`:

1. **Tier 1 — High-Frequency Lexical Stopwords (`word_weight = 2`)**:
   - Tokenizes text with a pre-compiled Unicode regex (`WORD_RE`) and performs $O(1)$ `frozenset` lookups against language-specific function words and helpdesk domain terms.
   - Ambiguous words shared across languages (e.g. `in`, `an`, `tag`, `problem`, `team`) are excluded from the stopword sets to prevent cross-lingual false positives.
2. **Tier 2 — Distinctive Characters (`char_weight = 3`)**:
   - Counts language-specific characters (e.g. German `ä, ö, ü, ß, Ä, Ö, Ü`; French `é, è, ê, ç, œ`; Spanish `ñ, ¿, ¡`).
3. **Tier 3 — Morphological Suffix Tie-Breaker (`suffix_weight = 1`)**:
   - Triggered only when the top two candidate languages are within a margin of $\le 2$ points (e.g., ultra-short technical subjects without stopwords).
   - Evaluates word endings such as German `-ung`, `-keit`, `-heit`, `-lich`, `-ieren` vs. English `-tion`, `-ment`, `-ing`, `-ed`, `-able`.
4. **Calibrated Confidence Scaling**:
   - Produces a bounded confidence score in $[0.5, 1.0]$ scaled by both the winning margin ($\frac{S_{\text{top}}}{S_{\text{top}} + S_{\text{second}}}$) and total evidence volume.

---

## 3. Extensibility: Adding New Languages Beyond English & German
The engine is decoupled from any hardcoded `en`/`de` branching via the `LanguageProfile` registry pattern.

### Option A: Enable Built-in Extended Profiles (`fr`, `es`)
`sentineldesk.language` ships with pre-configured profiles for English (`EN_PROFILE`), German (`DE_PROFILE`), French (`FR_PROFILE`), and Spanish (`ES_PROFILE`):

```python
from sentineldesk.language import (
    DE_PROFILE,
    EN_PROFILE,
    ES_PROFILE,
    FR_PROFILE,
    LanguageDetector,
)

detector = LanguageDetector(
    profiles=[EN_PROFILE, DE_PROFILE, FR_PROFILE, ES_PROFILE]
)
pred = detector.detect_with_confidence(
    subject="Problème de connexion",
    body="Bonjour, nous avons un problème avec notre compte depuis ce matin.",
)
print(pred.language, pred.confidence, pred.scores)
# -> 'fr', 0.9999, {'en': 0, 'de': 0, 'fr': 20, 'es': 0}
```

### Option B: Register a Custom Language Profile at Runtime
To add any new language (e.g., Italian `it` or Dutch `nl`), instantiate `LanguageProfile` and call `register_profile()`:

```python
import re
from sentineldesk.language import DE_PROFILE, EN_PROFILE, LanguageDetector, LanguageProfile

detector = LanguageDetector(profiles=[EN_PROFILE, DE_PROFILE])

it_profile = LanguageProfile(
    code="it",
    name="Italian",
    distinctive_chars=frozenset("àèéìòù"),
    stopwords=frozenset({
        "il",
        "lo",
        "gli",
        "della",
        "nella",
        "sono",
        "siamo",
        "abbiamo",
        "buongiorno",
        "grazie",
        "problema",
        "nostro",
        "sistema",
        "assistenza",
    }),
    suffix_pattern=re.compile(r"(?:zione|zioni|mento|menti)$"),
)
detector.register_profile(it_profile)

print(
    detector.detect(
        "Problema con il sistema", "Buongiorno, grazie per l'assistenza."
    )
)
# -> 'it'
```

---

## 4. Benchmark & Verification Results (61,765 Tickets)
Run the verification suite anytime via:
```bash
PYTHONPATH=. pytest -v tests/test_language.py
PYTHONPATH=. python3 scripts/verify_language_detector.py
```

| Metric | Result |
| :--- | :--- |
| **Total Tickets Tested** | `61,765` (49,412 Train/KB + 12,353 Holdout) |
| **Crashes on Missing `subject` / `body`** | `0` (5,299 missing subjects, 2 missing bodies) |
| **Average Latency per Ticket** | **~21 µs (`0.021 ms`)** |
| **Throughput** | **~48,000 tickets / sec** (`1.28 s` for all 61,765 rows) |
| **Agreement with `langdetect` (N=1,000 & N=2,000)** | **100.00%** (while running **~67× faster**) |
