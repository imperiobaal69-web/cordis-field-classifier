"""Model family, from a no-skill reference to frozen sentence embeddings.

All models expose the same two calls so the experiment runner, the evaluation
and the API never special-case a model:

    fit(train_texts, Y_train, val_texts, Y_val) -> self
    predict_proba(texts) -> float array (n_texts, n_labels)

Hyper-parameters are chosen on validation macro PR-AUC (threshold-free), never
on the test or shift sets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier

Texts = list[str]


class NotFittedError(RuntimeError):
    """A model was used for prediction before ``fit`` (or before its weights were loaded)."""


def _fitted[T](value: T | None, what: str) -> T:
    if value is None:
        raise NotFittedError(f"{what} is not fitted; call fit() or load a saved Classifier")
    return value


class Model(Protocol):
    name: str
    params: dict[str, object]

    def fit(self, texts: Texts, y: np.ndarray, val_texts: Texts, val_y: np.ndarray) -> Model: ...

    def predict_proba(self, texts: Texts) -> np.ndarray: ...


def _val_score(y: np.ndarray, proba: np.ndarray) -> float:
    present = y.sum(axis=0) > 0
    return float(average_precision_score(y[:, present], proba[:, present], average="macro"))


def _linear(c: float) -> OneVsRestClassifier:
    return OneVsRestClassifier(LogisticRegression(C=c, max_iter=2000), n_jobs=-1)


def _select_c(
    x: Any, y: np.ndarray, xv: Any, val_y: np.ndarray, grid: tuple[float, ...]
) -> tuple[float, OneVsRestClassifier, dict[float, float]]:
    """Fit one classifier per regularisation strength; keep the best on validation."""
    fitted = {c: _linear(c).fit(x, y) for c in grid}
    scores = {c: round(_val_score(val_y, clf.predict_proba(xv)), 4) for c, clf in fitted.items()}
    best = max(scores, key=lambda c: scores[c])
    return best, fitted[best], scores


class PriorModel:
    """Scores every project with the training label frequencies (no text used)."""

    name = "prior"

    def __init__(self) -> None:
        self.params: dict[str, object] = {}
        self.prior: np.ndarray | None = None

    def fit(self, texts: Texts, y: np.ndarray, val_texts: Texts, val_y: np.ndarray) -> PriorModel:
        self.prior = y.mean(axis=0)
        return self

    def predict_proba(self, texts: Texts) -> np.ndarray:
        return np.tile(_fitted(self.prior, "prior"), (len(texts), 1))


class TfidfLinear:
    """Word uni/bi-gram TF-IDF + one-vs-rest logistic regression."""

    name = "tfidf_logreg"

    def __init__(self, c_grid: tuple[float, ...] = (4.0, 16.0, 64.0)) -> None:
        self.c_grid = c_grid
        self.params: dict[str, object] = {}
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.5,
            max_features=300_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.clf: OneVsRestClassifier | None = None

    def fit(self, texts: Texts, y: np.ndarray, val_texts: Texts, val_y: np.ndarray) -> TfidfLinear:
        x = self.vectorizer.fit_transform(texts)
        xv = self.vectorizer.transform(val_texts)
        c_best, self.clf, scores = _select_c(x, y, xv, val_y, self.c_grid)
        self.params = {"C": c_best, "val_macro_pr_auc_by_C": scores, "vocab": x.shape[1]}
        return self

    def predict_proba(self, texts: Texts) -> np.ndarray:
        clf = _fitted(self.clf, self.name)
        return clf.predict_proba(self.vectorizer.transform(texts))


class Embedder:
    """Sentence-transformer encoder with an on-disk cache keyed by model and text."""

    def __init__(self, model_name: str, cache_dir: Path | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=_device())
        return self._model

    def encode(self, texts: Texts) -> np.ndarray:
        key = hashlib.sha256((self.model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[
            :16
        ]
        path = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if path and path.exists():
            return np.load(path)
        emb = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb


class EmbeddingLinear:
    """Frozen sentence embeddings + one-vs-rest logistic regression."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path | None = None,
        c_grid: tuple[float, ...] = (1.0, 4.0, 16.0),
    ) -> None:
        self.embedder = Embedder(model_name, cache_dir)
        self.name = "embed_logreg:" + model_name.split("/")[-1]
        self.c_grid = c_grid
        self.params: dict[str, object] = {"encoder": model_name}
        self.clf: OneVsRestClassifier | None = None

    def fit(
        self, texts: Texts, y: np.ndarray, val_texts: Texts, val_y: np.ndarray
    ) -> EmbeddingLinear:
        x, xv = self.embedder.encode(texts), self.embedder.encode(val_texts)
        c_best, self.clf, scores = _select_c(x, y, xv, val_y, self.c_grid)
        self.params |= {"C": c_best, "val_macro_pr_auc_by_C": scores}
        return self

    def predict_proba(self, texts: Texts) -> np.ndarray:
        clf = _fitted(self.clf, self.name)
        return clf.predict_proba(self.embedder.encode(texts))

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["embedder"] = Embedder(self.embedder.model_name, None, self.embedder.batch_size)
        return state


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
