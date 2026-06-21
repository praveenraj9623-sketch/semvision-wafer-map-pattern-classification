"""
evaluate.py — Standalone evaluation utilities (used by tests and UI).
"""
import os
import sys
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import STAGE2_CLASSES


def generate_classification_report(y_true, y_pred) -> dict:
    """Return sklearn classification_report as dict (target_names=STAGE2_CLASSES)."""
    return classification_report(
        y_true, y_pred,
        target_names=STAGE2_CLASSES,
        output_dict=True,
        zero_division=0,
    )


def compute_calibration_metrics(y_pred_proba: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute mean max-softmax for correct vs incorrect predictions.

    A well-calibrated model should show higher mean confidence on correct
    predictions than on incorrect ones.
    """
    max_sm       = np.max(y_pred_proba, axis=1)
    correct_mask = np.argmax(y_pred_proba, axis=1) == y_true
    return {
        "mean_max_softmax_correct":   float(max_sm[correct_mask].mean())
                                      if correct_mask.sum() > 0 else 0.0,
        "mean_max_softmax_incorrect": float(max_sm[~correct_mask].mean())
                                      if (~correct_mask).sum() > 0 else 0.0,
        "overall_mean_max_softmax":   float(max_sm.mean()),
    }


def find_worst_performing_class(per_class_metrics: dict):
    """Return (class_name, f1_score) for the class with lowest F1."""
    return min(per_class_metrics.items(), key=lambda x: x[1]["f1"])


def find_most_confused_pair(confusion_matrix: np.ndarray, class_names: list):
    """Return (class_a, class_b, count) for the most confused off-diagonal pair."""
    cm_copy = confusion_matrix.copy()
    np.fill_diagonal(cm_copy, 0)
    idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    return class_names[idx[0]], class_names[idx[1]], int(cm_copy[idx])
