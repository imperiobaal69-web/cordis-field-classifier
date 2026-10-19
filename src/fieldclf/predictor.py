"""The deployable unit: a fitted model + label vocabulary + tuned thresholds + metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from fieldclf.metrics import decide
from fieldclf.models import Model


@dataclass
class Classifier:
    model: Model
    labels: list[str]
    thresholds: np.ndarray
    meta: dict[str, object] = field(default_factory=dict)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)

    def predict(self, texts: list[str], top_k: int = 5) -> list[dict[str, object]]:
        proba = self.predict_proba(texts)
        chosen = decide(proba, self.thresholds)
        out: list[dict[str, object]] = []
        for p, c in zip(proba, chosen, strict=True):
            order = np.argsort(-p)
            out.append(
                {
                    "fields": [self.labels[j] for j in order if c[j]],
                    "scores": [
                        {
                            "field": self.labels[j],
                            "score": round(float(p[j]), 4),
                            "threshold": round(float(self.thresholds[j]), 2),
                        }
                        for j in order[:top_k]
                    ],
                }
            )
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # Models with framework-native weights (transformers) store them beside
        # the pickle instead of inside it.
        if hasattr(self.model, "export"):
            self.model.export(directory / "weights")
        joblib.dump(self, directory / "classifier.joblib", compress=3)
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str) + "\n")

    @staticmethod
    def load(directory: Path) -> Classifier:
        clf = joblib.load(directory / "classifier.joblib")
        if not isinstance(clf, Classifier):
            raise TypeError(f"{directory} does not contain a Classifier")
        if hasattr(clf.model, "attach"):
            clf.model.attach(directory / "weights")
        return clf
