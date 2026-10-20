"""Fine-tuning path, run end to end on a tiny random BERT (a few MB, CPU, seconds).

Skipped when the optional PyTorch stack is not installed; CI installs it in a
separate job.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from fieldclf.finetune import FineTuned, _fit_temperature, prior_logits
from fieldclf.models import NotFittedError
from fieldclf.predictor import Classifier

TINY = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="module")
def toy() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    words = [
        ["laser", "photon", "optics"],
        ["protein", "cell", "gene"],
        ["market", "trade", "firm"],
    ]
    texts, rows = [], []
    for _ in range(96):
        labels = rng.choice(3, size=rng.integers(1, 3), replace=False)
        texts.append(" ".join(w for j in labels for w in rng.choice(words[j], size=6)))
        rows.append([int(j in labels) for j in range(3)])
    return texts, np.array(rows)


@pytest.fixture(scope="module")
def fitted(toy: tuple[list[str], np.ndarray]) -> FineTuned:
    texts, y = toy
    model = FineTuned(TINY, max_length=24, epochs=2, batch_size=16, lr=1e-3, head_lr=1e-2)
    return model.fit(texts[:64], y[:64], texts[64:], y[64:])


def test_fit_records_history_and_temperature(fitted: FineTuned) -> None:
    history = fitted.params["history"]
    assert isinstance(history, list) and [h["epoch"] for h in history] == [1, 2]
    assert 0.5 <= fitted.temperature <= 3.0


def test_predictions_are_probabilities_of_the_right_shape(
    fitted: FineTuned, toy: tuple[list[str], np.ndarray]
) -> None:
    texts, _ = toy
    proba = fitted.predict_proba(texts[:5])
    assert proba.shape == (5, 3)
    assert ((proba > 0) & (proba < 1)).all()


def test_fixed_length_padding_gives_length_independent_scores(fitted: FineTuned) -> None:
    """A text scores the same alone and next to a much longer one (padding is masked)."""
    short = "laser photon"
    alone = fitted.predict_proba([short])
    batched = fitted.predict_proba([short, "protein cell gene " * 20])[:1]
    np.testing.assert_allclose(alone, batched, atol=1e-5)


def test_save_load_round_trip_is_exact(
    fitted: FineTuned, toy: tuple[list[str], np.ndarray], tmp_path
) -> None:
    texts, _ = toy
    Classifier(fitted, ["a", "b", "c"], np.full(3, 0.5)).save(tmp_path / "clf")
    loaded = Classifier.load(tmp_path / "clf")
    np.testing.assert_allclose(
        loaded.predict_proba(texts[:8]), fitted.predict_proba(texts[:8]), atol=1e-6
    )
    assert (tmp_path / "clf" / "weights" / "head.pt").exists()


def test_unfitted_model_raises() -> None:
    with pytest.raises(NotFittedError):
        FineTuned(TINY).predict_proba(["anything"])


def test_prior_logits_invert_to_label_frequencies() -> None:
    y = np.array([[1, 0], [1, 0], [0, 0], [1, 1]])
    np.testing.assert_allclose(1 / (1 + np.exp(-prior_logits(y))), [0.75, 0.25])


def test_temperature_recovers_known_scaling() -> None:
    rng = np.random.default_rng(1)
    true_logits = rng.normal(0, 2, size=(20_000, 1))
    y = (rng.uniform(size=true_logits.shape) < 1 / (1 + np.exp(-true_logits))).astype(int)
    # Logits twice as large as calibrated -> fitted temperature close to 2.
    assert abs(_fit_temperature(2 * true_logits, y) - 2.0) <= 0.1
