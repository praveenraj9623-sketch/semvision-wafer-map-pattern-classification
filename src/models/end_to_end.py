"""End-to-end evaluation of the Stage 1 gate plus Stage 2 classifier."""
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src.config_loader import STAGE2_CLASSES


SYSTEM_CLASS_NAMES = ["none", *STAGE2_CLASSES]


def evaluate_end_to_end(
    X,
    y_binary,
    y_pattern,
    splits_dict,
    stage1_model,
    stage2_model,
    stage1_threshold,
    confidence_threshold,
    persist=True,
):
    """Evaluate the deployed two-stage decision policy on global test data."""
    test_idx = np.asarray(splits_dict["global_test_idx"], dtype=int)
    X_test = X[test_idx]
    y_binary_test = np.asarray(y_binary)[test_idx]
    y_pattern_test = np.asarray(y_pattern)[test_idx]
    y_true_system = np.where(y_pattern_test == -1, 0, y_pattern_test + 1)

    gate_probability = stage1_model.predict_proba(
        X_test.reshape(len(X_test), -1)
    )[:, 1]
    gate_defect = gate_probability >= stage1_threshold

    stage2_prediction = np.full(len(test_idx), -1, dtype=int)
    stage2_confidence = np.zeros(len(test_idx), dtype=float)
    if gate_defect.any():
        probabilities = stage2_model.predict(X_test[gate_defect], verbose=0)
        stage2_prediction[gate_defect] = np.argmax(probabilities, axis=1)
        stage2_confidence[gate_defect] = np.max(probabilities, axis=1)

    forced_system_prediction = np.zeros(len(test_idx), dtype=int)
    forced_system_prediction[gate_defect] = stage2_prediction[gate_defect] + 1
    uncertain = gate_defect & (stage2_confidence < confidence_threshold)
    abstention_aware_prediction = forced_system_prediction.copy()
    abstention_aware_prediction[uncertain] = -1

    true_none = y_binary_test == 0
    true_defect = y_binary_test == 1
    correctly_identified_none = true_none & ~gate_defect
    normal_sent_to_stage2 = true_none & gate_defect
    defect_missed_by_gate = true_defect & ~gate_defect
    correct_defect_class = (
        true_defect
        & gate_defect
        & ~uncertain
        & (forced_system_prediction == y_true_system)
    )
    misclassified_defect = (
        true_defect
        & gate_defect
        & ~uncertain
        & (forced_system_prediction != y_true_system)
    )

    nine_class_confusion = confusion_matrix(
        y_true_system,
        forced_system_prediction,
        labels=list(range(len(SYSTEM_CLASS_NAMES))),
    )
    abstention_matrix = np.zeros(
        (len(SYSTEM_CLASS_NAMES), len(SYSTEM_CLASS_NAMES) + 1), dtype=int
    )
    for truth, prediction in zip(y_true_system, abstention_aware_prediction):
        column = len(SYSTEM_CLASS_NAMES) if prediction == -1 else int(prediction)
        abstention_matrix[int(truth), column] += 1

    macro_f1 = f1_score(
        y_true_system,
        abstention_aware_prediction,
        labels=list(range(len(SYSTEM_CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    defect_gate_recall = float(
        np.mean(gate_defect[true_defect]) if true_defect.any() else 0.0
    )

    results = {
        "evaluation_partition": "global_test",
        "global_test_samples": int(len(test_idx)),
        "stage1_decision_threshold": float(stage1_threshold),
        "stage2_confidence_threshold": float(confidence_threshold),
        "counts": {
            "correctly_identified_no_labelled_pattern": int(
                correctly_identified_none.sum()
            ),
            "no_labelled_pattern_sent_to_stage2": int(normal_sent_to_stage2.sum()),
            "defect_pattern_missed_by_stage1": int(defect_missed_by_gate.sum()),
            "correctly_classified_defect_pattern": int(correct_defect_class.sum()),
            "defect_pattern_sent_to_stage2_but_misclassified": int(
                misclassified_defect.sum()
            ),
            "uncertain_predictions": int(uncertain.sum()),
            "uncertain_true_defect": int((uncertain & true_defect).sum()),
            "uncertain_true_none": int((uncertain & true_none).sum()),
        },
        "end_to_end_macro_f1": float(macro_f1),
        "end_to_end_defect_recall": defect_gate_recall,
        "correct_accepted_defect_recall": float(
            correct_defect_class.sum() / true_defect.sum()
            if true_defect.any()
            else 0.0
        ),
        "abstention_rate": float(uncertain.mean()),
        "nine_class_names": SYSTEM_CLASS_NAMES,
        "nine_class_system_confusion_matrix": nine_class_confusion.tolist(),
        "nine_class_confusion_policy": (
            "For the required 9x9 matrix, uncertain Stage 2 cases retain their "
            "argmax class. The separate abstention-aware matrix records UNCERTAIN "
            "as a tenth prediction column."
        ),
        "abstention_aware_prediction_names": [
            *SYSTEM_CLASS_NAMES,
            "UNCERTAIN",
        ],
        "abstention_aware_confusion_matrix": abstention_matrix.tolist(),
    }

    if persist:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        reports_dir = os.path.join(project_root, "artifacts", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, "end_to_end_metrics.json")
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=2)
        print(f"End-to-end metrics saved -> {path}")
    return results
