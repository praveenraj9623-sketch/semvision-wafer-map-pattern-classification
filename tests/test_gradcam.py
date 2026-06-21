"""
test_gradcam.py — Verify Grad-CAM heatmaps in demo samples (requires completed training).
"""
import os
import sys
import pickle
import pytest
import numpy as np

pytestmark = pytest.mark.integration

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

_DEMO_PATH = os.path.join(_project_root, "data", "demo", "demo_samples.pkl")


@pytest.fixture(scope="module")
def demo_samples_with_gradcam():
    if not os.path.exists(_DEMO_PATH):
        pytest.skip("demo_samples.pkl not found — run integration training first")
    with open(_DEMO_PATH, "rb") as f:
        all_samples = pickle.load(f)
    samples = [s for s in all_samples if s.get("gradcam_heatmap") is not None]
    assert samples, "No demo samples with Grad-CAM heatmaps found"
    return samples


def test_demo_artifact_contract():
    if not os.path.exists(_DEMO_PATH):
        pytest.skip("demo_samples.pkl not found — run integration training first")
    assert os.path.getsize(_DEMO_PATH) < 10 * 1024 * 1024
    with open(_DEMO_PATH, "rb") as file_obj:
        samples = pickle.load(file_obj)
    defect_classes = {sample["true_class"] for sample in samples if sample["is_defect"]}
    expected = {"Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-Full", "Random", "Scratch"}
    assert defect_classes == expected
    assert any(not sample["is_defect"] for sample in samples)
    required = {
        "source_record_id", "raw_failureType", "encoded_label", "wafer_3ch",
        "split_name", "stage1_prediction", "stage2_prediction",
        "original_shape", "gradcam_class_idx",
    }
    for sample in samples:
        assert required.issubset(sample)
        assert sample["split_name"] == "test"
        expected = -1 if sample["true_class"] == "none" else [
            "Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Near-Full", "Random", "Scratch",
        ].index(sample["true_class"])
        assert sample["encoded_label"] == expected
        assert sample["gradcam_class_idx"] == sample["stage2_prediction"]


def test_gradcam_shape(demo_samples_with_gradcam):
    """Every Grad-CAM heatmap must have shape (64, 64)."""
    for s in demo_samples_with_gradcam:
        hm = s["gradcam_heatmap"]
        assert hm.shape == (64, 64), (
            f"Heatmap for class {s['true_class']} has shape {hm.shape}, expected (64, 64)"
        )


def test_gradcam_value_range(demo_samples_with_gradcam):
    """All heatmap values must be in [0.0, 1.0]."""
    for s in demo_samples_with_gradcam:
        hm = s["gradcam_heatmap"]
        assert hm.min() >= 0.0, f"Heatmap min {hm.min():.6f} < 0.0"
        assert hm.max() <= 1.0, f"Heatmap max {hm.max():.6f} > 1.0"


def test_gradcam_no_nan_inf(demo_samples_with_gradcam):
    """Heatmaps must not contain NaN or Inf values."""
    for s in demo_samples_with_gradcam:
        hm = s["gradcam_heatmap"]
        assert not np.any(np.isnan(hm)), f"NaN found in heatmap for class {s['true_class']}"
        assert not np.any(np.isinf(hm)), f"Inf found in heatmap for class {s['true_class']}"


def test_gradcam_not_all_zeros(demo_samples_with_gradcam):
    """Heatmaps should not be trivially all-zero (would indicate a Grad-CAM computation error)."""
    zero_count = sum(1 for s in demo_samples_with_gradcam if s["gradcam_heatmap"].max() < 1e-6)
    total = len(demo_samples_with_gradcam)
    # Allow up to 20% all-zero (degenerate cases with zero gradients can occur)
    assert zero_count / total < 0.20, (
        f"{zero_count}/{total} heatmaps are all-zero — Grad-CAM may not be working"
    )


if __name__ == "__main__":
    import sys
    if not os.path.exists(_DEMO_PATH):
        print("SKIP: demo_samples.pkl not found")
        sys.exit(0)
    with open(_DEMO_PATH, "rb") as f:
        samples = pickle.load(f)
    samples_with_hm = [s for s in samples if s.get("gradcam_heatmap") is not None]
    if not samples_with_hm:
        print("SKIP: no samples with Grad-CAM")
        sys.exit(0)
    test_gradcam_shape(samples_with_hm)
    test_gradcam_value_range(samples_with_hm)
    test_gradcam_no_nan_inf(samples_with_hm)
    test_gradcam_not_all_zeros(samples_with_hm)
    print(f"✅ test_gradcam.py passed ({len(samples_with_hm)} heatmaps verified)")
