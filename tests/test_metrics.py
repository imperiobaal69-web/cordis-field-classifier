import numpy as np

from fieldclf.metrics import decide, expected_calibration_error, summary, tune_thresholds


def test_decide_forces_one_label_when_none_passes():
    proba = np.array([[0.1, 0.3], [0.9, 0.2]])
    pred = decide(proba, np.array([0.5, 0.5]))
    assert pred.tolist() == [[0, 1], [1, 0]]
    assert decide(proba, np.array([0.5, 0.5]), at_least_one=False).tolist() == [[0, 0], [1, 0]]


def test_tune_thresholds_finds_the_separating_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(400, 1))
    proba = np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.02, size=y.shape)
    t = tune_thresholds(y, proba)
    assert 0.35 < t[0] < 0.65


def test_rare_labels_fall_back_to_global_threshold():
    y = np.zeros((100, 2), dtype=int)
    y[:50, 0] = 1
    y[0, 1] = 1  # one positive: too few for its own threshold
    proba = np.column_stack([y[:, 0] * 0.8 + 0.1, np.full(100, 0.5)])
    global_only = tune_thresholds(y, proba, min_positives=10_000)
    assert global_only[0] == global_only[1]
    t = tune_thresholds(y, proba, min_positives=20)
    assert t[1] == global_only[1]


def test_ece_zero_for_perfectly_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y.reshape(-1, 1), p.reshape(-1, 1)) < 0.01


def test_summary_perfect_prediction():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    m = summary(y, y.astype(float), y)
    assert m["micro_f1"] == m["macro_f1"] == m["macro_pr_auc"] == 1.0
    assert m["labels_per_project_pred"] == m["labels_per_project_true"]
