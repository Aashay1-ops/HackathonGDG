# Triage module (step ①)

Predicts `type`, `queue` and `priority` with calibrated confidence, plus a separate **critical score** (how likely the ticket is a high-priority Incident). The gate uses that score to escalate outages. No LLM is involved. The full spec is in `CLAUDE_1.md`.

## For teammates: how to use it

```python
from sentineldesk.triage.predict import Triage

triage = Triage.load("artifacts")          # Streamlit: wrap in @st.cache_resource
out = triage.predict(subject, body)        # subject may be None or ""
```

Each call returns:

| Key | Content | Used by |
|---|---|---|
| `type` / `queue` / `priority` | `label`, `confidence` and `probs` (in the fixed order from `artifacts/triage/labels.json`) | gate (low confidence), UI |
| `critical` | `score`, `threshold`, `fires`, `keyword_hits` | gate ("escalate to on-call") |
| `suggested_conf_thresholds` | one value per task, tuned on half B | gate owner decides whether to use them |
| `neighbours` | top 5 `train` tickets (id, cosine, labels) and the text "x of 20 similar tickets…" | UI explanation |
| `latency_ms` | time per stage | UI |
| `mode` | `"full"` (ensemble) or `"fallback"` (model 0 + temperature) | UI |

Keyword hits **never** change the labels. They are only inputs to the critical detector.

**For retrieval:** `scripts/embed.py` saves normalized embeddings at `artifacts/embeddings/<model>/{train,half_a,half_b}.npy`, with `<part>_ids.json` giving the ticket order. The kb parquet files in `artifacts/splits/` still include `answer`. The holdout `answer` is dropped in `split.py` and never written to disk.

## Train

```bash
python scripts/build_split.py          # split.load_split (team split); downloads data/tickets.csv if missing
python scripts/embed.py                # default intfloat/multilingual-e5-small
python scripts/train_triage.py         # add --fallback-only for model 0 + critical only
pytest -q tests/test_triage.py
python scripts/eval_triage.py          # holdout: ONCE, at the very end
```

Add `--sample 2000` to any script for a quick run. Sample runs go to `artifacts/sample_2000/` and never overwrite the full artifacts.

**On Colab (from VS Code):** use `notebooks/triage_colab.ipynb`. It clones the repo, trains on a T4 and sends the artifacts back as a zip. On a Mac with Apple silicon, the full run also works locally: embedding uses the GPU (MPS).

## Where each split part goes

| Part | Size | Used for |
|---|---|---|
| `train` | ~20.6k | TF-IDF, model 0, model 1, critical detector, neighbour index |
| `half_a` | ~1.1k | ensemble (6) |
| `half_b` | ~1.1k | temperature scaling, critical threshold (recall ≥ 0.90), suggested confidence thresholds |
| `test` | ~5.7k | `eval_triage.py` only. Labels are read only after predictions are saved |
