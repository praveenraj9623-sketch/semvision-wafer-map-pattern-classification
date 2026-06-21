"""Stage 1 binary gate training and recall-prioritized threshold selection."""
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import RANDOM_SEED, STAGE1_MAX_FALSE_POSITIVE_RATE


GATE_THRESHOLD_METHODOLOGY = (
    "On the validation partition, evaluate thresholds from 0.05 to 0.95 in "
    "0.01 increments. Among thresholds with false-positive rate <= "
    f"{STAGE1_MAX_FALSE_POSITIVE_RATE:.2f}, maximize defect-pattern recall; "
    "break ties by F1, precision, lower false-positive rate, then the higher "
    "threshold. Select the model using those validation operating points, "
    "prioritizing recall, PR-AUC, F1, and lower false-positive rate in that order."
)


def _binary_metrics(y_true, defect_probability, threshold):
    """Return gate metrics at one decision threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(defect_probability, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "defect_pattern_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "defect_pattern_recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_rate": float(false_positive_rate),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negative_count": int(tn),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "true_positive_count": int(tp),
    }


def select_gate_threshold(
    y_validation,
    defect_probability,
    max_false_positive_rate=STAGE1_MAX_FALSE_POSITIVE_RATE,
):
    """Select a recall-prioritized gate threshold using validation data only."""
    candidates = [
        _binary_metrics(y_validation, defect_probability, threshold)
        for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2)
    ]
    feasible = [
        item
        for item in candidates
        if item["false_positive_rate"] <= max_false_positive_rate
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda item: (
                item["defect_pattern_recall"],
                item["f1"],
                item["defect_pattern_precision"],
                -item["false_positive_rate"],
                item["threshold"],
            ),
        )
        constraint_satisfied = True
    else:
        selected = min(
            candidates,
            key=lambda item: (
                item["false_positive_rate"],
                -item["defect_pattern_recall"],
                -item["f1"],
            ),
        )
        constraint_satisfied = False
    return selected, candidates, constraint_satisfied


def _configure_mlflow(project_root):
    tracking_dir = Path(project_root, "artifacts", "mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())


def train_stage1(X, y_binary, splits_dict, class_distribution=None):
    """Train both gate candidates and select model/threshold on validation data."""
    np.random.seed(RANDOM_SEED)
    train_idx = splits_dict["stage1_train_idx"]
    val_idx = splits_dict["stage1_val_idx"]
    test_idx = splits_dict["stage1_test_idx"]

    flattened = X.reshape(X.shape[0], -1)
    X_train, y_train = flattened[train_idx], y_binary[train_idx]
    X_val, y_val = flattened[val_idx], y_binary[val_idx]
    X_test, y_test = flattened[test_idx], y_binary[test_idx]

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _configure_mlflow(project_root)
    mlflow.set_experiment("semvision_stage1")
    split_method = splits_dict.get("split_method", "stratified")
    fallback_reason = splits_dict.get("split_fallback_reason", "")

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_depth=15,
        ),
    }
    results = {}

    for model_name, model in candidates.items():
        print(f"Training Stage 1 candidate: {model_name}")
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, y_train)
            validation_probability = model.predict_proba(X_val)[:, 1]
            selected, threshold_table, constraint_satisfied = select_gate_threshold(
                y_val, validation_probability
            )
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "split_method": split_method,
                    "threshold_selection_partition": "validation",
                    "max_false_positive_rate": STAGE1_MAX_FALSE_POSITIVE_RATE,
                    "selected_threshold": selected["threshold"],
                }
            )
            mlflow.log_metrics(
                {
                    f"validation_{key}": value
                    for key, value in selected.items()
                    if isinstance(value, (int, float))
                }
            )
            results[model_name] = {
                "model": model,
                "run_id": run.info.run_id,
                "selected_threshold": selected["threshold"],
                "validation_metrics": selected,
                "threshold_candidates": threshold_table,
                "fpr_constraint_satisfied": constraint_satisfied,
            }

    champion_name = max(
        results,
        key=lambda name: (
            results[name]["validation_metrics"]["defect_pattern_recall"],
            results[name]["validation_metrics"]["pr_auc"],
            results[name]["validation_metrics"]["f1"],
            -results[name]["validation_metrics"]["false_positive_rate"],
        ),
    )
    champion_model = results[champion_name]["model"]
    champion_threshold = results[champion_name]["selected_threshold"]

    client = mlflow.tracking.MlflowClient()
    test_metrics_by_model = {}
    for model_name, result in results.items():
        test_probability = result["model"].predict_proba(X_test)[:, 1]
        test_metrics = _binary_metrics(
            y_test, test_probability, result["selected_threshold"]
        )
        test_metrics_by_model[model_name] = test_metrics
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float)):
                client.log_metric(
                    result["run_id"], f"test_{metric_name}", float(metric_value)
                )
        client.set_tag(
            result["run_id"], "champion", str(model_name == champion_name).lower()
        )

    champion_metrics = test_metrics_by_model[champion_name]
    models_dir = os.path.join(project_root, "artifacts", "models")
    reports_dir = os.path.join(project_root, "artifacts", "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    joblib.dump(champion_model, os.path.join(models_dir, "stage1_binary.pkl"))

    model_validation = {
        name: {
            "selected_threshold": result["selected_threshold"],
            "fpr_constraint_satisfied": result["fpr_constraint_satisfied"],
            "metrics": result["validation_metrics"],
        }
        for name, result in results.items()
    }
    full_metrics = {
        "champion": champion_name,
        "champion_selection_partition": "validation",
        "champion_selection_methodology": GATE_THRESHOLD_METHODOLOGY,
        "decision_threshold": float(champion_threshold),
        "threshold_selection": {
            "partition": "validation",
            "methodology": GATE_THRESHOLD_METHODOLOGY,
            "max_false_positive_rate": STAGE1_MAX_FALSE_POSITIVE_RATE,
            "constraint_satisfied": results[champion_name][
                "fpr_constraint_satisfied"
            ],
            "candidates": results[champion_name]["threshold_candidates"],
        },
        "champion_validation_metrics": results[champion_name][
            "validation_metrics"
        ],
        "champion_metrics": champion_metrics,
        "logistic_regression_metrics": test_metrics_by_model[
            "logistic_regression"
        ],
        "random_forest_metrics": test_metrics_by_model["random_forest"],
        "model_validation": model_validation,
        "split_method": split_method,
        "split_fallback_reason": fallback_reason or None,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "total_samples": int(len(y_binary)),
        "class_distribution": class_distribution or {},
    }
    with open(
        os.path.join(reports_dir, "stage1_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(full_metrics, file_obj, indent=2)

    print(
        f"Stage 1 champion: {champion_name}, threshold={champion_threshold:.2f}, "
        f"validation recall={results[champion_name]['validation_metrics']['recall']:.4f}, "
        f"test PR-AUC={champion_metrics['pr_auc']:.4f}"
    )
    return champion_model, full_metrics
