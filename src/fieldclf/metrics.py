"""Multi-label evaluation: threshold-free ranking quality, thresholded F1, calibration."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def tune_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid: np.ndarray | None = None,
    min_positives: int = 20,
) -> np.ndarray:
    """Per-label threshold maximising F1 on validation data.

    Labels with fewer than ``min_positives`` validation examples fall back to the
    single global threshold that maximises micro-F1, because a per-label optimum
    estimated from a handful of positives is mostly noise.
    """
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    global_f1 = [f1_score(y_true, proba >= t, average="micro", zero_division=0) for t in grid]
    global_t = _best(grid, global_f1)
    thresholds = np.full(y_true.shape[1], global_t)
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() < min_positives:
            continue
        scores = [f1_score(y_true[:, j], proba[:, j] >= t, zero_division=0) for t in grid]
        thresholds[j] = _best(grid, scores)
    return thresholds


def _best(grid: np.ndarray, scores: list[float]) -> float:
    """Middle of the arg-max plateau: when several thresholds tie, the centre is the
    one least sensitive to small score shifts (an arg-max would pick the lowest)."""
    s = np.asarray(scores)
    ties = np.flatnonzero(s >= s.max() - 1e-12)
    return float(grid[ties[len(ties) // 2]])


def decide(proba: np.ndarray, thresholds: np.ndarray, at_least_one: bool = True) -> np.ndarray:
    """Apply thresholds; optionally force the top-scoring label when none passes.

    Every project in the data has at least one field, so an empty prediction is
    always wrong; forcing the arg-max converts it into a guess that is often right.
    """
    pred = (proba >= thresholds).astype(np.int8)
    if at_least_one:
        empty = pred.sum(axis=1) == 0
        pred[empty, proba[empty].argmax(axis=1)] = 1
    return pred


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    """ECE over all (project, label) pairs, equal-width bins."""
    p, y = proba.ravel(), y_true.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def reliability_curve(
    y_true: np.ndarray, proba: np.ndarray, bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, y = proba.ravel(), y_true.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    conf, acc, count = [], [], []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            conf.append(p[mask].mean())
            acc.append(y[mask].mean())
            count.append(int(mask.sum()))
    return np.array(conf), np.array(acc), np.array(count)


def _macro_pr_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    present = y_true.sum(axis=0) > 0
    return float(average_precision_score(y_true[:, present], proba[:, present], average="macro"))


def summary(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Headline metrics. Macro averages skip labels absent from ``y_true``."""
    present = y_true.sum(axis=0) > 0
    return {
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_f1": float(
            f1_score(y_true[:, present], pred[:, present], average="macro", zero_division=0)
        ),
        "samples_f1": float(f1_score(y_true, pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, pred, average="micro", zero_division=0)),
        "macro_pr_auc": _macro_pr_auc(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "labels_per_project_true": float(y_true.sum(axis=1).mean()),
        "labels_per_project_pred": float(pred.sum(axis=1).mean()),
        "n": len(y_true),
    }


def per_label(
    y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray, vocab: list[str]
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for j, name in enumerate(vocab):
        support = int(y_true[:, j].sum())
        rows.append(
            {
                "label": name,
                "support": support,
                "precision": float(precision_score(y_true[:, j], pred[:, j], zero_division=0)),
                "recall": float(recall_score(y_true[:, j], pred[:, j], zero_division=0)),
                "f1": float(f1_score(y_true[:, j], pred[:, j], zero_division=0)),
                "pr_auc": float(average_precision_score(y_true[:, j], proba[:, j]))
                if support
                else float("nan"),
            }
        )
    return rows


def bootstrap_f1(
    y_true: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Bootstrap distribution of micro- and macro-F1 over projects.

    Projects are resampled with replacement (as multinomial weights, so each
    replicate costs one matrix product). Macro-F1 averages the labels present
    in ``y_true``, matching :func:`summary`. Passing the same ``seed`` for two
    models evaluated on the same projects gives paired replicates, so their
    difference has a valid bootstrap distribution too.
    """
    present = y_true.sum(axis=0) > 0
    yt, yp = y_true[:, present].astype(np.int64), pred[:, present].astype(np.int64)
    tp, fp, fn = yt * yp, (1 - yt) * yp, yt * (1 - yp)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(yt), np.full(len(yt), 1 / len(yt)), size=n_boot).astype(np.float64)
    tp_l, fp_l, fn_l = w @ tp, w @ fp, w @ fn
    micro = 2 * tp_l.sum(1) / np.maximum(2 * tp_l.sum(1) + fp_l.sum(1) + fn_l.sum(1), 1)
    per_label = 2 * tp_l / np.maximum(2 * tp_l + fp_l + fn_l, 1)
    return {"micro_f1": micro, "macro_f1": per_label.mean(axis=1)}


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution."""
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
