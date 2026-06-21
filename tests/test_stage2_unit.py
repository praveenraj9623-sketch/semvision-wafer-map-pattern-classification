import numpy as np

from models.stage2_cnn import (
    compute_calibration_metrics,
    select_abstention_threshold,
)


def test_abstention_candidates_and_selection_contract():
    y_true = np.tile(np.arange(8), 20)
    probabilities = np.full((len(y_true), 8), 0.02, dtype=float)
    probabilities[np.arange(len(y_true)), y_true] = np.linspace(0.55, 0.95, len(y_true))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    result = select_abstention_threshold(probabilities, y_true)
    assert [item["threshold"] for item in result["candidates"]] == [
        round(value, 2) for value in np.arange(0.30, 0.901, 0.05)
    ]
    assert result["selection_partition"] == "validation"
    assert 0.30 <= result["selected_threshold"] <= 0.90
    for candidate in result["candidates"]:
        assert 0.0 <= candidate["coverage"] <= 1.0
        assert 0.0 <= candidate["uncertainty_rate"] <= 1.0
        assert 0.0 <= candidate["error_rate"] <= 1.0


def test_calibration_metrics_are_finite_and_binned():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]
    ])
    metrics = compute_calibration_metrics(probabilities, y_true, n_bins=5)
    assert np.isfinite(metrics["expected_calibration_error"])
    assert np.isfinite(metrics["multiclass_brier_score"])
    assert len(metrics["reliability_bins"]) == 5

