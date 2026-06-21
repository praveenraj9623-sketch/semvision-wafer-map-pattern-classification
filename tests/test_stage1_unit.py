import numpy as np

from models.stage1_model import _binary_metrics, select_gate_threshold


def test_gate_metrics_include_required_operating_characteristics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.60, 0.40, 0.80, 0.95])
    metrics = _binary_metrics(y_true, probabilities, threshold=0.5)
    for key in (
        "roc_auc", "pr_auc", "defect_pattern_recall",
        "defect_pattern_precision", "false_negative_rate",
        "false_positive_rate", "f1",
    ):
        assert key in metrics


def test_gate_threshold_is_selected_from_validation_candidates():
    y_true = np.array([0] * 10 + [1] * 10)
    probabilities = np.array(
        [0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.22, 0.60]
        + [0.20, 0.30, 0.40, 0.55, 0.65, 0.70, 0.75, 0.80, 0.90, 0.98]
    )
    selected, candidates, constraint_satisfied = select_gate_threshold(
        y_true, probabilities, max_false_positive_rate=0.20
    )
    assert candidates
    assert constraint_satisfied
    assert selected["false_positive_rate"] <= 0.20
    assert selected["defect_pattern_recall"] >= 0.8

