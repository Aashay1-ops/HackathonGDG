"""Model 6: per task, a small LR stacked on log-probs of models 0, 1, 2 (trained on half A)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from sentineldesk.triage import TASKS
from sentineldesk.triage.calibration import safe_log
from sentineldesk.triage.labels import align_proba


def stack_features(prob_sets: list[dict[str, np.ndarray]], task: str) -> np.ndarray:
    return np.hstack([safe_log(p[task]) for p in prob_sets])


class Ensemble:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.clfs: dict[str, LogisticRegression] = {}

    def fit(self, prob_sets: list[dict[str, np.ndarray]], y: dict[str, np.ndarray],
            labels: dict[str, list[str]]) -> "Ensemble":
        for t in TASKS:
            clf = LogisticRegression(max_iter=2000, C=self.C, random_state=42)
            clf.fit(stack_features(prob_sets, t), np.asarray(labels[t])[y[t]])
            self.clfs[t] = clf
        return self

    def predict_proba(self, prob_sets: list[dict[str, np.ndarray]], labels: dict[str, list[str]]) -> dict[str, np.ndarray]:
        return {t: align_proba(c, c.predict_proba(stack_features(prob_sets, t)), labels[t]) for t, c in self.clfs.items()}
