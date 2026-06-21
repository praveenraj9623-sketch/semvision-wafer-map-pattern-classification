"""
prepare_demo.py — Precompute demo samples for the Streamlit app.

Run ONCE after training completes. Output: data/demo/demo_samples.pkl (<10 MB).
The Streamlit app must NEVER load LSWMD.pkl or data/processed/X.npy directly.

Selected samples: up to 5 per defect class (≤40) + 5 normal wafers.
Each sample dict contains:
  - wafer_3ch         : (64, 64, 3) float32
  - true_class        : string
  - is_defect         : int 0|1
  - stage1_prediction : int
  - stage1_confidence : float
  - stage2_prediction : int or None
  - stage2_confidence : float or None
  - stage2_confidences: list[float] (all 8 classes) or None
  - predicted_class   : str or None
  - gradcam_heatmap   : (64, 64) float32 or None
  - sample_number     : int (1-based within class)
"""
import os
import sys
import pickle
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import STAGE2_CLASSES, NONE_CLASS, DEFAULT_CONFIDENCE_THRESHOLD


def prepare_demo_samples() -> list:
    """Prepare and save demo_samples.pkl. Returns the list of sample dicts."""
    project_root  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(project_root, "data", "processed")
    models_dir    = os.path.join(project_root, "artifacts", "models")
    demo_dir      = os.path.join(project_root, "data", "demo")
    os.makedirs(demo_dir, exist_ok=True)

    # ── Load processed data ──────────────────────────────────────────────────
    print("Loading processed data …")
    X         = np.load(os.path.join(processed_dir, "X.npy"))
    labels    = np.load(os.path.join(processed_dir, "labels.npz"), allow_pickle=True)
    y_binary  = labels["y_binary"]
    y_pattern = labels["y_pattern"]
    required_provenance = {
        "source_indices", "original_shapes", "raw_failure_types"
    }
    missing_provenance = required_provenance.difference(labels.files)
    if missing_provenance:
        raise ValueError(
            "labels.npz predates provenance support; rerun the full training "
            f"pipeline. Missing: {sorted(missing_provenance)}"
        )
    source_indices = labels["source_indices"]
    original_shapes = labels["original_shapes"]
    raw_failure_types = labels["raw_failure_types"]

    splits     = np.load(os.path.join(processed_dir, "splits.npz"), allow_pickle=True)
    test_idx = splits["global_test_idx"]

    reports_dir = os.path.join(project_root, "artifacts", "reports")
    with open(os.path.join(reports_dir, "stage1_metrics.json"), encoding="utf-8") as f:
        stage1_metrics = json.load(f)
    with open(os.path.join(reports_dir, "stage2_metrics.json"), encoding="utf-8") as f:
        stage2_metrics = json.load(f)
    stage1_threshold = float(stage1_metrics["decision_threshold"])
    confidence_threshold = float(
        stage2_metrics.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    )

    # ── Load trained models ──────────────────────────────────────────────────
    import joblib
    import tensorflow as tf
    from visualization.gradcam import compute_gradcam

    stage1_model = joblib.load(os.path.join(models_dir, "stage1_binary.pkl"))
    stage2_model = tf.keras.models.load_model(
        os.path.join(models_dir, "stage2_pattern_cnn.h5"), compile=False
    )

    # ── Identify test-set indices by category ────────────────────────────────
    defect_test_idx = test_idx[y_pattern[test_idx] != -1]
    none_test_idx = test_idx[y_pattern[test_idx] == -1]
    test_index_set = set(test_idx.tolist())

    demo_samples = []

    # ── Helper: build one sample dict ────────────────────────────────────────
    def _build_sample(idx, true_class, sample_number):
        wafer_3ch = X[idx]
        encoded_label = int(y_pattern[idx])
        expected_class = (
            NONE_CLASS if encoded_label == -1 else STAGE2_CLASSES[encoded_label]
        )
        assert true_class == expected_class, (
            f"Demo label mismatch at processed index {idx}: "
            f"display={true_class}, encoded={expected_class}"
        )
        assert idx in test_index_set, "Demo sample is not in global test split"

        # Stage 1
        flat             = wafer_3ch.reshape(1, -1)
        s1_prob          = stage1_model.predict_proba(flat)[0]
        s1_pred          = int(s1_prob[1] >= stage1_threshold)
        s1_conf          = float(s1_prob[s1_pred])
        s1_defect_prob   = float(s1_prob[1])

        # Precompute Stage 2 for every sample so the UI can follow the Stage 1
        # gate decision without consulting the ground-truth label.
        inp            = wafer_3ch[np.newaxis, ...]
        s2_proba       = stage2_model.predict(inp, verbose=0)[0]
        s2_pred        = int(np.argmax(s2_proba))
        s2_conf        = float(s2_proba[s2_pred])
        s2_confs       = [float(p) for p in s2_proba]
        pred_class     = STAGE2_CLASSES[s2_pred]
        heatmap        = compute_gradcam(stage2_model, inp, s2_pred)

        source_record = source_indices[idx]
        if isinstance(source_record, np.generic):
            source_record = source_record.item()

        return {
            "index":               int(idx),
            "processed_position":  int(idx),
            "source_record_id":    source_record,
            "raw_failureType":     raw_failure_types[idx],
            "encoded_label":       encoded_label,
            "original_shape":      [int(v) for v in original_shapes[idx]],
            "split_name":          "test",
            "wafer_3ch":           wafer_3ch,
            "true_class":          true_class,
            "is_defect":           int(y_binary[idx]),
            "stage1_prediction":   s1_pred,
            "stage1_confidence":   s1_conf,
            "stage1_defect_probability": s1_defect_prob,
            "stage1_decision_threshold": stage1_threshold,
            "stage2_prediction":   s2_pred,
            "stage2_confidence":   s2_conf,
            "stage2_confidences":  s2_confs,
            "predicted_class":     pred_class,
            "stage2_uncertain":    bool(s2_conf < confidence_threshold),
            "confidence_threshold": confidence_threshold,
            "gradcam_heatmap":     heatmap,
            "gradcam_class_idx":   s2_pred,
            "sample_number":       sample_number,
        }

    # ── Select 5 per defect class ────────────────────────────────────────────
    for cls_idx, cls_name in enumerate(STAGE2_CLASSES):
        cls_mask    = y_pattern[defect_test_idx] == cls_idx
        cls_indices = defect_test_idx[cls_mask]
        n_select    = min(5, len(cls_indices))
        if n_select == 0:
            print(f"  ⚠ No test samples found for class: {cls_name}")
            continue
        for i, raw_idx in enumerate(cls_indices[:n_select]):
            print(f"  Building {cls_name} sample {i+1}/{n_select} …", end="\r")
            demo_samples.append(_build_sample(int(raw_idx), cls_name, i + 1))
        print(f"  {cls_name}: {n_select} samples done              ")

    # ── Select 5 normal wafers ────────────────────────────────────────────────
    n_none = min(5, len(none_test_idx))
    for i, raw_idx in enumerate(none_test_idx[:n_none]):
        demo_samples.append(_build_sample(int(raw_idx), NONE_CLASS, i + 1))
    print(f"  none: {n_none} samples done")

    # ── Save ─────────────────────────────────────────────────────────────────
    demo_path = os.path.join(demo_dir, "demo_samples.pkl")
    with open(demo_path, "wb") as f:
        pickle.dump(demo_samples, f, protocol=4)

    size_mb = os.path.getsize(demo_path) / (1024 ** 2)
    print(f"\nDemo samples saved → {demo_path}")
    print(f"  {len(demo_samples)} samples,  {size_mb:.2f} MB")
    if size_mb > 10:
        raise RuntimeError(f"Demo artifact is {size_mb:.2f} MB; required maximum is 10 MB")

    return demo_samples


if __name__ == "__main__":
    print("Preparing demo samples …")
    samples = prepare_demo_samples()
    print(f"Done. {len(samples)} demo samples prepared.")
