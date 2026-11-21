"""Model 2: similarity-weighted votes of the top-k most similar `train` tickets (no training).

Uses FAISS IndexFlatIP when installed; otherwise an exact NumPy inner product (same results).
"""
from __future__ import annotations

import logging

import numpy as np

from sentineldesk.triage import TASKS

log = logging.getLogger(__name__)


class NeighbourIndex:
    def __init__(self, emb: np.ndarray, ids: list[str], codes: dict[str, np.ndarray],
                 labels: dict[str, list[str]], k: int = 20, p: float = 1.0, alpha: float = 0.1):
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.ids = list(ids)
        self.id_set = set(self.ids)
        self.codes, self.labels = codes, labels
        self.k, self.p, self.alpha = k, p, alpha
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        except ImportError:
            log.info("faiss not installed: using NumPy exact search")
            self.index = None

    def _search(self, Q: np.ndarray, kk: int) -> tuple[np.ndarray, np.ndarray]:
        kk = min(kk, len(self.ids))
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if self.index is not None:
            return self.index.search(Q, kk)
        S = Q @ self.emb.T
        I = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        Ssub = np.take_along_axis(S, I, axis=1)
        order = np.argsort(-Ssub, axis=1)
        return np.take_along_axis(Ssub, order, axis=1), np.take_along_axis(I, order, axis=1)

    def search(self, Q: np.ndarray, query_ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (sims, idx) per query, with each query's own ticket removed if it is in the index."""
        need_excl = query_ids is not None and any(q in self.id_set for q in query_ids)
        S, I = self._search(Q, self.k + 1 if need_excl else self.k)
        if not need_excl:
            return S, I
        S_out = np.zeros((len(Q), self.k), dtype=np.float32)
        I_out = np.zeros((len(Q), self.k), dtype=np.int64)
        for r, qid in enumerate(query_ids):
            keep = [j for j in range(I.shape[1]) if self.ids[I[r, j]] != qid][: self.k]
            S_out[r], I_out[r] = S[r, keep], I[r, keep]
        return S_out, I_out

    def vote(self, S: np.ndarray, I: np.ndarray) -> dict[str, np.ndarray]:
        w = np.maximum(S, 0.0) ** self.p
        out = {}
        for t in TASKS:
            n_cls = len(self.labels[t])
            scores = np.full((len(S), n_cls), self.alpha, dtype=np.float64)
            nb = self.codes[t][I]  # (n, k) class codes of neighbours
            for c in range(n_cls):
                scores[:, c] += (w * (nb == c)).sum(axis=1)
            out[t] = scores / scores.sum(axis=1, keepdims=True)
        return out

    def describe(self, S_row: np.ndarray, I_row: np.ndarray, top_n: int = 5) -> dict:
        """UI payload for one ticket: top neighbours + "x of k similar tickets ..." per task."""
        top = []
        for s, i in zip(S_row[:top_n], I_row[:top_n]):
            row = {"ticket_id": self.ids[i], "similarity": round(float(s), 4)}
            row.update({t: self.labels[t][self.codes[t][i]] for t in TASKS})
            top.append(row)
        k = len(I_row)
        expl = {}
        for t in TASKS:
            counts = np.bincount(self.codes[t][I_row], minlength=len(self.labels[t]))
            c = int(counts.argmax())
            lab = self.labels[t][c]
            verb = "went to" if t == "queue" else "were"
            expl[t] = f"{counts[c]} of {k} similar tickets {verb} {lab}"
        return {"top": top, "explanation": expl}
