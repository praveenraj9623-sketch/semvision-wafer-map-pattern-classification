"""
stage2_cnn.py — 8-class defect pattern CNN for Stage 2.

AUGMENTATION ABLATION:
  Variant A — no augmentation (baseline)
  Variant B — rotation-only augmentation (0/90/180/270 + h-flip + v-flip)
  Both trained with identical architecture and hyperparameters.
  Winner selected by val macro-F1. If difference < 0.01, prefer simpler model.

AUGMENTATION NOTE:
  Pixel values are CATEGORICAL (0=missing, 1=pass, 2=fail channels as floats).
  Never use brightness/contrast/blur — only spatial transforms that preserve
  the integer channel semantics.

Per-class metrics use sklearn's classification_report for correctness.
class_weights are passed to model.fit as required by spec.
Training history is saved to artifacts/reports/training_history.json.
"""
import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    balanced_accuracy_score, classification_report
)
import mlflow
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import (
    NUM_DEFECT_CLASSES, CNN_EPOCHS, CNN_PATIENCE, CNN_BATCH_SIZE,
    RANDOM_SEED, STAGE2_CLASSES, MIN_ABSTENTION_COVERAGE, CALIBRATION_BINS,
)


ABSTENTION_SELECTION_METHODOLOGY = (
    "Evaluate max-softmax thresholds from 0.30 to 0.90 in 0.05 increments "
    f"on the validation partition. Among candidates with coverage >= "
    f"{MIN_ABSTENTION_COVERAGE:.0%}, select the highest accepted-sample macro "
    "F1; break ties by lower accepted-sample error rate, higher coverage, then "
    "the higher threshold. Test data is not used for threshold selection."
)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn() -> keras.Model:
    """Build the Stage-2 CNN exactly as specified."""
    model = keras.Sequential(
        [
            layers.Input(shape=(64, 64, 3)),
            layers.Conv2D(32,  (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64,  (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(NUM_DEFECT_CLASSES, activation="softmax"),
        ],
        name="stage2_cnn",
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _augment_single(img, label):
    """
    Spatial augmentation for a SINGLE wafer image tensor.
    Applied via tf.data.Dataset.map — runs in graph mode.

    Allowed: rotation by 0/90/180/270 degrees, h-flip, v-flip.
    NOT allowed: brightness/contrast/blur (pixel values are categorical).
    """
    # Random rotation: k = 0,1,2,3 → 0°, 90°, 180°, 270°
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)

    # Random horizontal flip with p=0.5
    img = tf.image.random_flip_left_right(img)

    # Random vertical flip with p=0.5
    img = tf.image.random_flip_up_down(img)

    return img, label


class _MLflowEpochLogger(keras.callbacks.Callback):
    """Record Keras epoch metrics on the active MLflow run."""

    def on_epoch_end(self, epoch, logs=None):
        clean_logs = {
            str(name): float(value)
            for name, value in (logs or {}).items()
            if value is not None
        }
        if clean_logs:
            mlflow.log_metrics(clean_logs, step=int(epoch))


def compute_calibration_metrics(probabilities, y_true, n_bins=CALIBRATION_BINS):
    """Compute top-label ECE, multiclass Brier score, and reliability bins."""
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    expected_calibration_error = 0.0
    reliability_bins = []

    for bin_index in range(n_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        if bin_index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        average_confidence = float(confidence[mask].mean()) if count else 0.0
        accuracy = float(correct[mask].mean()) if count else 0.0
        if count:
            expected_calibration_error += (
                count / len(y_true) * abs(accuracy - average_confidence)
            )
        reliability_bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "average_confidence": average_confidence,
                "accuracy": accuracy,
            }
        )

    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    brier_score = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "expected_calibration_error": float(expected_calibration_error),
        "multiclass_brier_score": brier_score,
        "n_bins": int(n_bins),
        "reliability_bins": reliability_bins,
    }


def select_abstention_threshold(probabilities, y_validation):
    """Select the deployed Stage 2 abstention threshold on validation data."""
    probabilities = np.asarray(probabilities, dtype=float)
    y_validation = np.asarray(y_validation, dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    candidates = []
    for threshold in np.round(np.arange(0.30, 0.901, 0.05), 2):
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        coverage = float(accepted.mean())
        if accepted_count:
            accepted_macro_f1 = float(
                f1_score(
                    y_validation[accepted],
                    predictions[accepted],
                    labels=list(range(NUM_DEFECT_CLASSES)),
                    average="macro",
                    zero_division=0,
                )
            )
            accepted_error_rate = float(
                np.mean(predictions[accepted] != y_validation[accepted])
            )
        else:
            accepted_macro_f1 = 0.0
            accepted_error_rate = 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accepted_sample_macro_f1": accepted_macro_f1,
                "uncertainty_rate": float(1.0 - coverage),
                "error_rate": accepted_error_rate,
                "accepted_count": accepted_count,
                "total_count": int(len(y_validation)),
            }
        )

    eligible = [
        item for item in candidates if item["coverage"] >= MIN_ABSTENTION_COVERAGE
    ]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda item: (
            item["accepted_sample_macro_f1"],
            -item["error_rate"],
            item["coverage"],
            item["threshold"],
        ),
    )
    return {
        "selected_threshold": selected["threshold"],
        "selection_partition": "validation",
        "selection_methodology": ABSTENTION_SELECTION_METHODOLOGY,
        "minimum_coverage": MIN_ABSTENTION_COVERAGE,
        "minimum_coverage_satisfied": bool(eligible),
        "selected_candidate": selected,
        "candidates": candidates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_cnn(
    model, X_test, y_test, measure_latency=True, predicted_probabilities=None
) -> dict:
    """
    Full evaluation of the CNN on the provided test set.

    Computes:
      - accuracy, macro_f1, weighted_f1, balanced_accuracy
      - per-class precision, recall, F1, support (from classification_report)
      - confusion matrix
      - worst class (lowest F1)
      - worst confusion pair (highest off-diagonal cell)
      - calibrated confidence: mean max-softmax for correct vs incorrect preds
      - inference latency (average over 100 single-sample predictions)
    """
    y_pred_proba = (
        np.asarray(predicted_probabilities)
        if predicted_probabilities is not None
        else model.predict(X_test, verbose=0)
    )
    y_pred       = np.argmax(y_pred_proba, axis=1)

    accuracy      = float(accuracy_score(y_test, y_pred))
    macro_f1      = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
    weighted_f1   = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    balanced_acc  = float(balanced_accuracy_score(y_test, y_pred))

    # ── Per-class metrics via classification_report ──────────────────────────
    report_dict = classification_report(
        y_test, y_pred,
        labels=list(range(NUM_DEFECT_CLASSES)),
        target_names=STAGE2_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = {}
    for cls in STAGE2_CLASSES:
        row = report_dict.get(cls, {})
        per_class_metrics[cls] = {
            "precision": float(row.get("precision", 0.0)),
            "recall":    float(row.get("recall",    0.0)),
            "f1":        float(row.get("f1-score",  0.0)),
            "support":   int(row.get("support",     0)),
        }

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_DEFECT_CLASSES)))

    # ── Worst class ──────────────────────────────────────────────────────────
    worst_class = min(per_class_metrics.keys(),
                      key=lambda k: per_class_metrics[k]["f1"])

    # ── Worst confusion pair (off-diagonal max) ──────────────────────────────
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    pair_idx   = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
    worst_pair = [
        STAGE2_CLASSES[pair_idx[0]],
        STAGE2_CLASSES[pair_idx[1]],
        int(cm_no_diag[pair_idx]),
    ]

    # ── Calibration ──────────────────────────────────────────────────────────
    correct_mask   = y_pred == y_test
    incorrect_mask = ~correct_mask
    max_sm         = np.max(y_pred_proba, axis=1)
    cal_correct    = float(max_sm[correct_mask].mean())   if correct_mask.sum()   > 0 else 0.0
    cal_incorrect  = float(max_sm[incorrect_mask].mean()) if incorrect_mask.sum() > 0 else 0.0
    calibration = compute_calibration_metrics(y_pred_proba, y_test)

    # ── Inference latency ────────────────────────────────────────────────────
    latency_ms = 0.0
    if measure_latency:
        sample = X_test[:1]
        model.predict(sample, verbose=0)
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict(sample, verbose=0)
        latency_ms = float((time.perf_counter() - t0) / 100 * 1000)

    return {
        "accuracy":              accuracy,
        "macro_f1":              macro_f1,
        "weighted_f1":           weighted_f1,
        "balanced_accuracy":     balanced_acc,
        "per_class_metrics":     per_class_metrics,
        "worst_class":           worst_class,
        "worst_confusion_pair":  worst_pair,
        "confusion_matrix":      cm.tolist(),
        "max_softmax_correct":   cal_correct,
        "max_softmax_incorrect": cal_incorrect,
        "confidence_separation": {
            "mean_max_softmax_correct": cal_correct,
            "mean_max_softmax_incorrect": cal_incorrect,
        },
        "calibration": calibration,
        "expected_calibration_error": calibration["expected_calibration_error"],
        "multiclass_brier_score": calibration["multiclass_brier_score"],
        "inference_latency_ms":  latency_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_variant(
    X_train, y_train, X_val, y_val,
    class_weight_dict, use_augmentation=False, run_name="cnn"
):
    """
    Train one CNN variant and return (model, history_dict, val_macro_f1).
    """
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Build tf.data pipeline ───────────────────────────────────────────────
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=min(len(y_train), 4096), seed=RANDOM_SEED)
        .batch(CNN_BATCH_SIZE)
    )
    if use_augmentation:
        train_ds = train_ds.unbatch().map(
            _augment_single, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(CNN_BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(CNN_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CNN_PATIENCE,
            restore_best_weights=True,
        ),
        _MLflowEpochLogger(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CNN_EPOCHS,
        class_weight=class_weight_dict,   # ← spec requires this
        callbacks=callbacks,
        verbose=1,
    )

    # Val macro-F1 for ablation comparison
    val_pred  = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_macro = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

    history_dict = {
        "loss":         [float(v) for v in history.history["loss"]],
        "val_loss":     [float(v) for v in history.history["val_loss"]],
        "accuracy":     [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "checkpoint_metric": "val_loss",
        "checkpoint_mode": "min",
        "restore_best_weights": True,
        "epochs_ran": int(len(history.history["loss"])),
    }

    return model, history_dict, val_macro


def train_stage2_cnn(X, y_pattern, splits_dict):
    """
    Train Stage 2 CNN with augmentation ablation.

    Returns
    -------
    best_model         : keras.Model
    test_metrics       : dict (full evaluation on test set)
    augmentation_winner: str  ('with_augmentation' | 'no_augmentation')
    """
    np.random.seed(RANDOM_SEED)

    train_idx = splits_dict["stage2_train_idx"]
    val_idx   = splits_dict["stage2_val_idx"]
    test_idx  = splits_dict["stage2_test_idx"]

    X_train, y_train = X[train_idx], y_pattern[train_idx]
    X_val,   y_val   = X[val_idx],   y_pattern[val_idx]
    X_test,  y_test  = X[test_idx],  y_pattern[test_idx]

    # ── Class weights ────────────────────────────────────────────────────────
    classes        = np.unique(y_train)
    missing_classes = np.setdiff1d(np.arange(NUM_DEFECT_CLASSES), classes)
    if missing_classes.size:
        names = [STAGE2_CLASSES[int(i)] for i in missing_classes]
        raise ValueError(f"Stage 2 training split is missing classes: {names}")
    class_weights  = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    tracking_dir = Path(project_root, "artifacts", "mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("semvision_stage2")

    # ── Variant A: no augmentation ───────────────────────────────────────────
    print("\n[Stage 2] Training Variant A (no augmentation) …")
    with mlflow.start_run(run_name="cnn_no_augmentation"):
        model_a, history_a, val_f1_a = _train_variant(
            X_train, y_train, X_val, y_val, class_weight_dict,
            use_augmentation=False
        )
        mlflow.log_params(dict(variant="A", augmentation=False,
                               epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
                               patience=CNN_PATIENCE))
        mlflow.log_metric("val_macro_f1", val_f1_a)
    print(f"  Variant A val macro-F1 = {val_f1_a:.4f}")

    # ── Variant B: with augmentation ─────────────────────────────────────────
    print("[Stage 2] Training Variant B (with augmentation) …")
    with mlflow.start_run(run_name="cnn_augmentation"):
        model_b, history_b, val_f1_b = _train_variant(
            X_train, y_train, X_val, y_val, class_weight_dict,
            use_augmentation=True
        )
        mlflow.log_params(dict(variant="B", augmentation=True,
                               epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
                               patience=CNN_PATIENCE))
        mlflow.log_metric("val_macro_f1", val_f1_b)
    print(f"  Variant B val macro-F1 = {val_f1_b:.4f}")

    # ── Ablation result ──────────────────────────────────────────────────────
    diff = val_f1_b - val_f1_a
    if diff >= 0.01:
        best_model         = model_b
        best_history       = history_b
        augmentation_winner = "with_augmentation"
        print(f"  Winner: Variant B (augmentation) +{diff:.4f}")
    else:
        best_model         = model_a
        best_history       = history_a
        augmentation_winner = "no_augmentation"
        note = f"difference = {diff:.4f} < 0.01 → prefer simpler model"
        print(f"  Winner: Variant A (no augmentation): {note}")

    if abs(diff) < 0.01:
        ablation_note = "Absolute validation macro-F1 difference < 0.01; simpler model preferred."
    elif diff < 0:
        ablation_note = "The no-augmentation baseline outperformed the augmented variant."
    else:
        ablation_note = "The augmented variant improved validation macro-F1 by at least 0.01."

    ablation_result = {
        "variant_a_val_macro_f1": val_f1_a,
        "variant_b_val_macro_f1": val_f1_b,
        "difference":             diff,
        "winner":                 augmentation_winner,
        "note":                   ablation_note,
    }

    # ── Final test evaluation ────────────────────────────────────────────────
    validation_probabilities = best_model.predict(X_val, verbose=0)
    abstention_validation = select_abstention_threshold(
        validation_probabilities, y_val
    )
    confidence_threshold = abstention_validation["selected_threshold"]
    print(
        f"[Stage 2] Validation-selected confidence threshold: "
        f"{confidence_threshold:.2f}"
    )
    print("[Stage 2] Evaluating best model on test set …")
    test_metrics = _evaluate_cnn(best_model, X_test, y_test)
    test_metrics["augmentation_ablation"] = ablation_result
    test_metrics["confidence_threshold"] = confidence_threshold
    test_metrics["abstention_validation"] = abstention_validation

    with mlflow.start_run(run_name=f"stage2_final_{augmentation_winner}"):
        mlflow.log_params(dict(augmentation_winner=augmentation_winner,
                               confidence_threshold=confidence_threshold,
                               threshold_selection_partition="validation"))
        top_level_metrics = {
            key: value for key, value in test_metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        mlflow.log_metrics(top_level_metrics)
        for class_name, class_metrics in test_metrics["per_class_metrics"].items():
            slug = class_name.lower().replace("-", "_")
            mlflow.log_metrics({
                f"{slug}_{metric_name}": float(metric_value)
                for metric_name, metric_value in class_metrics.items()
            })
        mlflow.set_tags({
            "worst_class": test_metrics["worst_class"],
            "worst_confusion_true": test_metrics["worst_confusion_pair"][0],
            "worst_confusion_predicted": test_metrics["worst_confusion_pair"][1],
        })
        mlflow.log_dict(test_metrics, "stage2_metrics.json")

    # ── Save best model ──────────────────────────────────────────────────────
    models_dir = os.path.join(project_root, "artifacts", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "stage2_pattern_cnn.h5")
    best_model.save(model_path)
    print(f"Best model saved → {model_path}")

    # ── Save reports ─────────────────────────────────────────────────────────
    reports_dir = os.path.join(project_root, "artifacts", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    metrics_path = os.path.join(reports_dir, "stage2_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Stage 2 metrics saved → {metrics_path}")

    ablation_path = os.path.join(reports_dir, "augmentation_ablation.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_result, f, indent=2)
    print(f"Ablation result saved → {ablation_path}")

    # ── Save training history ────────────────────────────────────────────────
    history_path = os.path.join(reports_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(best_history, f, indent=2)
    print(f"Training history saved → {history_path}")

    return best_model, test_metrics, augmentation_winner
