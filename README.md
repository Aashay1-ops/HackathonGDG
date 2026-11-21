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
| Classifier | embeddings + logistic regression / fine-tuned transformer | | |
| Vector store | FAISS / Chroma / numpy | | |
| UI | Streamlit / Gradio / other | | |
| What happens after "Escalate" | | | |
| What the AI draft must contain | | | |
| Confidence and similarity thresholds | | | |
| Team roles (5 people) | | | |
