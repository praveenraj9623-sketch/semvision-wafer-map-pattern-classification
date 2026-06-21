"""
test_stage1.py — Sanity check Stage 1 artifacts (requires completed training).
"""
import os
import sys
import json
import pytest

pytestmark = pytest.mark.integration

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)


def test_stage1_roc_auc():
    """Stage 1 champion ROC-AUC must exceed 0.70."""
    metrics_path = os.path.join(_project_root, "artifacts", "reports", "stage1_metrics.json")
    if not os.path.exists(metrics_path):
        pytest.skip("stage1_metrics.json not found — run integration training first")

    with open(metrics_path) as f:
        metrics = json.load(f)

    roc_auc = metrics["champion_metrics"]["roc_auc"]
    assert roc_auc > 0.70, (
        f"Stage 1 ROC-AUC {roc_auc:.4f} is below sanity threshold 0.70. "
        "Check model training or data quality."
    )
    print(f"✅ Stage 1 ROC-AUC = {roc_auc:.4f}")


def test_stage1_metrics_structure():
    """stage1_metrics.json must have required keys."""
    metrics_path = os.path.join(_project_root, "artifacts", "reports", "stage1_metrics.json")
    if not os.path.exists(metrics_path):
        pytest.skip("stage1_metrics.json not found — run integration training first")

    with open(metrics_path) as f:
        metrics = json.load(f)

    required_keys = [
        "champion", "champion_metrics",
        "logistic_regression_metrics", "random_forest_metrics",
        "split_method",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"

    metric_keys = [
        "roc_auc", "pr_auc", "defect_pattern_recall",
        "defect_pattern_precision", "false_negative_rate",
        "false_positive_rate", "f1",
    ]
    for key in metric_keys:
        assert key in metrics["champion_metrics"], f"champion_metrics missing: {key}"
    assert metrics["champion_selection_partition"] == "validation"
    assert 0.0 < metrics["decision_threshold"] < 1.0
    assert metrics["threshold_selection"]["candidates"]


if __name__ == "__main__":
    test_stage1_roc_auc()
    test_stage1_metrics_structure()
    print("✅ test_stage1.py passed")
