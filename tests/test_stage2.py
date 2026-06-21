"""
test_stage2.py — Sanity checks for Stage 2 artifacts (requires completed training).
"""
import os
import sys
import json
import pytest

pytestmark = pytest.mark.integration

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))


def test_stage2_macro_f1():
    """Stage 2 macro F1 must exceed 0.30 (realistic floor for 8-class imbalanced)."""
    path = os.path.join(_project_root, "artifacts", "reports", "stage2_metrics.json")
    if not os.path.exists(path):
        pytest.skip("stage2_metrics.json not found — run integration training first")

    with open(path) as f:
        metrics = json.load(f)

    macro_f1 = metrics["macro_f1"]
    assert macro_f1 > 0.30, (
        f"Stage 2 macro F1 {macro_f1:.4f} is below sanity floor 0.30. "
        "8-class imbalanced — check data split and class weights."
    )
    print(f"✅ Stage 2 macro F1 = {macro_f1:.4f}")


def test_stage2_metrics_structure():
    """stage2_metrics.json must have all required fields."""
    path = os.path.join(_project_root, "artifacts", "reports", "stage2_metrics.json")
    if not os.path.exists(path):
        pytest.skip("stage2_metrics.json not found — run integration training first")

    with open(path) as f:
        metrics = json.load(f)

    required = [
        "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
        "per_class_metrics", "worst_class", "worst_confusion_pair",
        "confusion_matrix", "max_softmax_correct", "max_softmax_incorrect",
        "inference_latency_ms",
        "confidence_threshold", "abstention_validation", "calibration",
    ]
    for key in required:
        assert key in metrics, f"stage2_metrics.json missing key: {key}"

    # per_class_metrics must have precision, recall, f1, support per class
    from config_loader import STAGE2_CLASSES
    for cls in STAGE2_CLASSES:
        if cls in metrics["per_class_metrics"]:
            cm = metrics["per_class_metrics"][cls]
            for subkey in ("precision", "recall", "f1", "support"):
                assert subkey in cm, f"per_class_metrics[{cls}] missing: {subkey}"


def test_model_card_exists_and_complete():
    """model_card.json must exist and contain all 13 required sections."""
    path = os.path.join(_project_root, "artifacts", "reports", "model_card.json")
    if not os.path.exists(path):
        pytest.skip("model_card.json not found — run integration training first")

    with open(path) as f:
        mc = json.load(f)

    required_sections = [
        "model_name", "model_type", "domain_note", "intended_use",
        "out_of_scope", "training_data", "data_split_method",
        "augmentation_used", "performance", "per_class_f1",
        "limitations", "random_seed", "inference_latency_ms",
    ]
    for section in required_sections:
        assert section in mc, f"model_card.json missing section: '{section}'"

    print("✅ Model card contains all 13 required sections")


def test_training_history_records_restored_checkpoint():
    path = os.path.join(_project_root, "artifacts", "reports", "training_history.json")
    if not os.path.exists(path):
        pytest.skip("training_history.json not found — run integration training first")
    with open(path) as file_obj:
        history = json.load(file_obj)
    assert history["restore_best_weights"] is True
    assert history["checkpoint_metric"] == "val_loss"
    assert 1 <= history["best_epoch"] <= history["epochs_ran"]


if __name__ == "__main__":
    test_stage2_macro_f1()
    test_stage2_metrics_structure()
    test_model_card_exists_and_complete()
    print("✅ test_stage2.py passed")
