# SentinelDesk

Hackathon project for the GDG Cloud Boston × Northeastern hackathon.

SentinelDesk takes an incoming support ticket (English or German) and:

1. works out what the ticket is and how urgent it is,
2. finds similar tickets we resolved in the past,
3. decides whether the AI may answer or a human must take it.

## The problem

Helpdesks lose time in two ways:

- **Missed outages.** A critical outage (for example, a VPN failure that cuts hospital staff off from patient records) lands in the same inbox as "what does your product cost?" and waits hours.
- **Made-up answers.** Generic AI chatbots write confident replies that promise refunds, SLAs or features nobody agreed to.

Our goal is to **never miss an outage and never make up an answer.**

## Agreed workflow

Steps ①–③ are decided. Everything else is still open (see [Open decisions](#open-decisions)).

```
   👤 Customer sends a ticket (subject may be empty + body, en or de)
                        │
                        ▼
┌───────────────────────────────────────────────┐
│  ① UNDERSTAND IT                               │
│  • What language? (en / de)                    │
│  • What kind?   Incident / Request /           │
│                 Problem / Change               │
│  • Which team?  Tech Support, Billing,         │
│                 IT, Sales, ...                 │
│  • How urgent?  high / medium / low            │
│  • How sure are we? (confidence %)             │
└───────────────────────┬───────────────────────┘
                        ▼
┌───────────────────────────────────────────────┐
│  ② LOOK UP THE PAST                            │
│  "Have we solved something like this before?"  │
│  → find 3-5 most similar resolved tickets      │
│    and how agents answered them                │
└───────────────────────┬───────────────────────┘
                        ▼
┌───────────────────────────────────────────────┐
│  ③ SAFETY CHECK: is it OK for AI to answer?   │
│                                                │
│   Escalate to a human if ANY is true:          │
│   ✗ urgent outage (high-priority Incident)     │
│   ✗ system isn't confident in its triage       │
│   ✗ no similar past ticket exists              │
│   ✗ customer asks for something we've never    │
│     done (refund promise, new feature, ...)    │
└───────────┬───────────────────────┬───────────┘
        risky│                       │safe
            ▼                       ▼
      🔴 Escalate              🟢 AI may draft
      (details TBD)            (details TBD)
```

## Architecture (hybrid)

We use classical ML where results must be measurable, rules where we can't afford to miss something, and the LLM only to write the reply. The LLM's output is then checked before anyone sees it.

```
 Ticket (subject?, body)
        │
        ▼
 ① TRIAGE
   ├─ ML: multilingual embeddings → LogReg ×3 (type / queue / priority)
   │      + CalibratedClassifierCV → calibrated confidence
   └─ Rules: outage keywords in en+de ("down", "ausgefallen", "kein Zugriff", ...)
          → force priority=high, type=Incident   (protects high-priority Incident recall)
        │
        ▼
 ② RETRIEVAL (hybrid search)
   ├─ BM25 (exact terms: product names, error codes)
   ├─ Dense multilingual embeddings (meaning; German ↔ English)
   └─ Reciprocal Rank Fusion → top 3–5
        │
        ▼
 ③ GATE (deterministic rules, each with a reason string)
   high-priority Incident │ low confidence │ low similarity │ unsupported ask
        │                                         │
     escalate                                   safe
        │                                         ▼
        │                     ④ DRAFT: LLM, given only the retrieved answers
        │                        returns JSON {reply, cited_ids, unsupported_asks}
        │                                         ▼
        │                     ⑤ VERIFY (rules)
        │                        • cited_ids ⊆ retrieved ticket_ids
        │                        • every number / € / $ / % / SLA term ("within X days")
        │                          in the reply must also appear in a retrieved answer
        │                        • keep placeholders like <name>, <acc_num> as they are
        │                        • check fails → escalate instead of showing the draft
        ▼                                         ▼
  🔴 Escalate to On-Call (reason)        🟢 Auto-Draft Ready
```

### Design rules

- **The LLM does not do triage.** LLM confidence scores aren't calibrated, and LLM calls would make the holdout evaluation (about 5.7k tickets) slow, rate-limited and non-deterministic.
- **The UI shows the dense cosine similarity** and the "no precedent" threshold uses it. The RRF score is only used to order results, because it is not a similarity.
- **Fit everything on the 80% split only:** BM25 IDF statistics, the embedding index, the classifiers and calibration.
- **Build the outage keyword list from the 80% split** (its `Outage`/`Disruption` tags) and from common sense. Never tune it by looking at holdout errors.
- **Report two sets of metrics:** "ML only" and "ML + rules" (Macro-F1 and high-priority Incident recall). The rules raise recall and may cost some precision, and we show that trade-off on purpose.
- **Mask PII before any external LLM call** (bonus): emails, phone numbers, IPs and account numbers.

## RAG plan

**Knowledge base:** one entry per past ticket from the 80% split: `ticket_id`, subject + body, answer, language, type, queue, priority, tags. Rows with an empty answer are dropped.

**What we search on:** subject + body. Answers are not searchable, because we match on the customer's problem, not on how an agent phrased the fix.

**Using categories.** A new ticket arrives with no category, only our classifier's prediction (type is right about 87% of the time, queue only about 50%), so:
- ✅ **Soft boost (default).** After searching, move up past tickets whose type/queue matches the prediction, by an amount scaled by the classifier's confidence. A low-confidence guess barely changes anything.
- ✅ **Categories go to the LLM.** Each precedent is passed with its queue/type so the LLM can judge how relevant it is.
- 🧪 **Experiment:** add tags/categories to the past tickets' searchable text (helps keyword search). Keep it only if it measurably helps.
- ❌ **No hard filter by predicted queue.** It would hide the right match about half the time.

**Pipeline order:** classify, then search (so the boost can use the predictions), then the safety check.

**How we evaluate retrieval:**
- Self-retrieval: a past ticket searched by its own text must come back first.
- How often the top 5 share the holdout ticket's type and queue. This is only a sanity check, because the boost inflates it by design.
- Hand-check about 20 holdout tickets: is the top answer actually useful?
- Cross-lingual hit rate, and search latency p50/p95.

**Handoff to the LLM:** `[{ticket_id, similarity, language, type, queue, answer}]`, top 5.

## Required deliverables (from the problem statement)

- [ ] **Triage:** predict type, queue and priority, each with a confidence score
- [ ] **Metrics on the 20% holdout split:** Macro-F1, and recall on high-priority Incidents
- [ ] **Escalation gate:** explicit rules, and the reason for every escalation is shown
- [ ] **Retrieval:** top 3–5 similar past tickets, shown with `ticket_id`, similarity score and answer snippet
- [ ] **Draft reply:** in the customer's language, built only from past answers, citing the `ticket_id`s it used, and never inventing prices, SLAs or legal promises
- [ ] **Live console:** judges paste a ticket and see the predictions, confidence, latency, escalation badge, past tickets and draft
- [ ] **Submission:** public git repo with `requirements.txt` and an evaluation script, plus a README and a 5-minute live demo

**Bonus features (only if there's time):** cross-lingual matching (English past tickets for German input and vice versa), outage burst detection, PII masking.

## Dataset

- Hugging Face: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC-BY-NC-4.0, about 61.8k tickets)
- **Inputs:** `subject` (can be empty, so never crash on it) and `body`
- **Targets:** `type`, `queue`, `priority`
- **Language:** detect it from the text at demo time. Don't read the `language` column.

### ⚠️ Leakage rules (breaking these can get us disqualified)

- Use one fixed split for everything: shuffle with `random_state=42`, first 80% = knowledge base + training data, last 20% = holdout.
- Never index or read the holdout split's `answer`, `type`, `queue` or `priority`, except to score predictions at the very end.
- Fit everything (vectorizers, models, the search index) on the 80% split only.

## Open decisions

| Topic | Options | Owner | Decision |
|---|---|---|---|
| LLM for drafting | Gemini (check the hackathon benefits page) / other | | |
| Embedding model | multilingual-e5 / bge-m3 / TF-IDF | | |
| Classifier | embeddings + logistic regression / fine-tuned transformer | | ✅ embeddings + LogReg + calibration, plus outage rules |
| Retrieval | dense only / hybrid | | ✅ hybrid: BM25 + dense, fused with RRF |
| Vector store | FAISS / Chroma / numpy | | |
| UI | Streamlit / Gradio / other | | |
| What happens after "Escalate" | | | |
| What the AI draft must contain | | | |
| Confidence and similarity thresholds | | | |
| Team roles (5 people) | | | |
