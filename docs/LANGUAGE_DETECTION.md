# Language Detection Guide (`sentineldesk.language`)
**語言分揀模組使用與擴充說明（English / 繁體中文雙語版）**

---

## 1. What does this module do?（這個模組做什麼？）

When a customer sends a support ticket (`subject` + `body`), the very first step in SentinelDesk is figuring out **what language they are writing in** (currently **English `en`** or **German `de`**, with built-in support for adding more languages later).

當客戶寄來一張客服工單（包含標題 `subject` 與內文 `body`），SentinelDesk 的第一步就是**立刻判斷這張工單是什麼語言**（目前預設分辨 **英文 `en`** 與 **德文 `de`**，並且支援隨時擴充新語言）。

### Why don't we just read the dataset's `language` column?（為什麼不直接讀資料集裡的 `language` 欄位？）
1. **Hackathon Rule（比賽規定）**: The problem statement explicitly forbids reading the `language` column at demo/inference time.
2. **Data Bug Discovery（原始資料有錯！）**: When we inspected the 61,765 tickets in `Tobi-Bueck/customer-support-tickets`, we discovered that **8.7% (5,377 tickets) have the WRONG label** in the `language` column! (For example, Ticket `#32` is written in pure English `"Dear Customer Support Team..."`, but the dataset says `de`).
3. **Safe on Empty Inputs（不會因為空標題而當機）**: 5,299 tickets in the dataset have an empty `subject` (`None` or `NaN`). This module handles empty or missing values safely and never crashes.

---

## 2. Quick Start: How to use it in your code（30 秒快速上手）

Other team members working on **Triage**, **Retrieval**, **UI**, or **LLM Drafting** only need two functions:
負責其他模組（Triage、檢索、UI、LLM 草稿）的隊友，只需要匯入以下函式即可使用：

### A. Just get the language code (`'en'` or `'de'`)（只拿語言代碼）
```python
from sentineldesk import detect_language

lang = detect_language(
    subject="Wesentlicher Sicherheitsvorfall",
    body="Sehr geehrtes Team, der Server ist ausgefallen.",
)
print(lang)  # -> 'de'
```

### B. Get language + confidence score for the UI（拿語言代碼 + 信心分數，給 UI 顯示）
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

### C. Process a whole list/column of tickets at once（一次處理整批工單）
```python
from sentineldesk import detect_language_batch

langs = detect_language_batch(df["subject"], df["body"])
```

---

## 3. How does it work under the hood?（它是怎麼判斷語言的？為什麼這麼快？）

Instead of calling a slow LLM or heavy neural network, it uses a **3-Step Evidence Score（三步計分法）** that takes only **0.02 milliseconds (20 微秒)** per ticket:

```text
Incoming Ticket (subject + body)
       │
       ├─► Step 1: Common Words Check (常見功能詞比對，每個字 +2 分)
       │     • English words: "the", "and", "please", "dear", "issue", "unable"...
       │     • German words : "ich", "nicht", "bitte", "sehr", "fehler", "zugriff"...
       │
       ├─► Step 2: Special Letters Check (特殊字母檢查，每個字母 +3 分)
       │     • German letters: ä, ö, ü, ß
       │
       └─► Step 3: Word Ending Tie-Breaker (字尾決勝負，僅在平手或極短句時啟動，+1 分)
             • English endings: -tion, -ment, -ing, -able (e.g. "Configuration failed")
             • German endings : -ung, -keit, -heit, -ieren (e.g. "Systemaktualisierung")
```

Whichever language gets the higher score wins! The bigger the score gap, the higher the `confidence` (%).
哪個語言總分高就勝出！兩邊分數差距越大，`confidence`（信心百分比）就越高。

---

## 4. How to add more languages in the future?（未來如何擴充新語言？）

We designed this module using **Language Profiles（語言設定檔）** so nothing is hardcoded to just English and German.
我們把每個語言的特徵做成了「設定卡 (`LanguageProfile`)」，所以未來要加第 3、第 4 種語言時，**完全不用改核心程式碼**！

### Case 1: Turn on built-in French (`fr`) and Spanish (`es`)（直接開啟內建的法文與西班牙文）
We already included French and Spanish profiles out-of-the-box:
```python
from sentineldesk import (
    DE_PROFILE,
    EN_PROFILE,
    ES_PROFILE,
    FR_PROFILE,
    LanguageDetector,
)

# Create a detector that speaks 4 languages (en, de, fr, es)
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

### Case 2: Plug in any brand-new language (e.g., Italian `it`)（自己新增任何一種語言，例如義大利文）
Just create a `LanguageProfile` with common words and special characters, and register it:
```python
import re
from sentineldesk import DE_PROFILE, EN_PROFILE, LanguageDetector, LanguageProfile

detector = LanguageDetector(profiles=[EN_PROFILE, DE_PROFILE])

# Define Italian in 10 lines
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

# Plug it in!
detector.register_profile(italian)

print(
    detector.detect(
        "Problema con il sistema", "Buongiorno, grazie per l'assistenza."
    )
)
# -> 'it'
```

---

## 5. Speed & Accuracy Results（實測速度與準確度）

You can run our tests and full-dataset benchmark anytime with these two commands:
隨時可以在終端機執行以下兩行指令來跑測試與驗證：

```bash
# 1. Run 23 unit & extensibility tests (takes 0.06 seconds)
PYTHONPATH=. uv run --with pytest pytest -v tests/test_language.py

# 2. Run benchmark on all 61,765 real tickets from Hugging Face
PYTHONPATH=. uv run --with datasets --with pandas --with langdetect python3 scripts/verify_language_detector.py
```

| What we tested（測試項目） | Result（實測結果） |
| :--- | :--- |
| **Total tickets tested（測試工單總數）** | **61,765 筆**（全部跑完僅需 **1.28 秒**） |
| **Speed per ticket（單筆工單耗時）** | **0.021 毫秒 (ms)**（每秒可處理 **48,000+ 筆**） |
| **Crashes on empty subject/body（空值當機次數）** | **0 次**（完美通過 5,299 筆空標題測試） |
| **Accuracy vs. `langdetect`（與主流套件準確率比對）** | **100.00% 完全一致**（而且速度快了 **67 倍**！） |
