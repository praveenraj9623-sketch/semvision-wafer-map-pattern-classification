"""
test_split.py — Unit tests for make_splits (no dataset needed).

Tests with 200 mock samples:
  - stage1_test_idx has no overlap with stage1_train_idx
  - stage2_test_idx has no overlap with stage2_train_idx
  - val_idx sets have no overlap with their respective test_idx
  - All indices are valid (within range of N)
"""
import os
import sys
import pytest
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from data.split import make_splits


@pytest.fixture(scope="module")
def mock_splits():
    """Create mock dataset and run make_splits once for all tests."""
    np.random.seed(42)
    N = 200

    X         = np.random.rand(N, 64, 64, 3).astype(np.float32)
    y_binary  = np.array([0] * 50 + [1] * 150)   # 50 normal, 150 defect
    y_pattern = np.array([-1] * 50 + list(np.tile(range(8), 19))[:150])
    lot_groups = np.array([f"lot_{i % 25}" for i in range(N)])

    return make_splits(X, y_binary, y_pattern, lot_groups, persist=False)


def test_stage1_no_train_test_overlap(mock_splits):
    train = set(mock_splits["stage1_train_idx"].tolist())
    test  = set(mock_splits["stage1_test_idx"].tolist())
    assert len(train & test) == 0, "Stage 1 train and test overlap!"


def test_stage1_no_val_test_overlap(mock_splits):
    val  = set(mock_splits["stage1_val_idx"].tolist())
    test = set(mock_splits["stage1_test_idx"].tolist())
    assert len(val & test) == 0, "Stage 1 val and test overlap!"


def test_stage1_no_train_val_overlap(mock_splits):
    train = set(mock_splits["stage1_train_idx"].tolist())
    val   = set(mock_splits["stage1_val_idx"].tolist())
    assert len(train & val) == 0, "Stage 1 train and val overlap!"


def test_stage2_no_train_test_overlap(mock_splits):
    train = set(mock_splits["stage2_train_idx"].tolist())
    test  = set(mock_splits["stage2_test_idx"].tolist())
    assert len(train & test) == 0, "Stage 2 train and test overlap!"


def test_stage2_no_val_test_overlap(mock_splits):
    val  = set(mock_splits["stage2_val_idx"].tolist())
    test = set(mock_splits["stage2_test_idx"].tolist())
    assert len(val & test) == 0, "Stage 2 val and test overlap!"


def test_all_indices_in_range(mock_splits):
    """All indices must be valid (0..199)."""
    N = 200
    for key in ["stage1_train_idx", "stage1_val_idx", "stage1_test_idx",
                "stage2_train_idx", "stage2_val_idx", "stage2_test_idx"]:
        idx = mock_splits[key]
        assert idx.min() >= 0,   f"{key} has negative index"
        assert idx.max() < N,    f"{key} has index >= N={N}"


def test_lot_groups_do_not_leak(mock_splits):
    lot_groups = np.array([f"lot_{i % 25}" for i in range(200)])
    for prefix in ("stage1", "stage2"):
        train_lots = set(lot_groups[mock_splits[f"{prefix}_train_idx"]])
        val_lots = set(lot_groups[mock_splits[f"{prefix}_val_idx"]])
        test_lots = set(lot_groups[mock_splits[f"{prefix}_test_idx"]])
        assert not (train_lots & val_lots)
        assert not (train_lots & test_lots)
        assert not (val_lots & test_lots)


def test_stage_splits_are_derived_from_one_global_partition(mock_splits):
    y_pattern = np.array([-1] * 50 + list(np.tile(range(8), 19))[:150])
    for split_name in ("train", "val", "test"):
        global_idx = mock_splits[f"global_{split_name}_idx"]
        assert np.array_equal(
            mock_splits[f"stage1_{split_name}_idx"], global_idx
        )
        assert np.array_equal(
            mock_splits[f"stage2_{split_name}_idx"],
            global_idx[y_pattern[global_idx] != -1],
        )


def test_grouped_split_falls_back_with_exact_reason_when_coverage_is_impossible():
    samples_per_class = 30
    global_labels = np.repeat(np.arange(9), samples_per_class)
    y_pattern = np.where(global_labels == 0, -1, global_labels - 1)
    y_binary = (global_labels != 0).astype(int)
    X = np.zeros((len(global_labels), 1), dtype=np.float32)
    groups = np.empty(len(global_labels), dtype=object)
    groups[global_labels == 0] = "single_none_lot"
    defect_positions = np.flatnonzero(global_labels != 0)
    groups[defect_positions] = [
        f"defect_lot_{index % 25}" for index in range(len(defect_positions))
    ]

    splits = make_splits(X, y_binary, y_pattern, groups, persist=False)
    assert splits["split_method"] == "stratified"
    assert "preserved all 9 classes" in splits["split_fallback_reason"]


if __name__ == "__main__":
    splits = mock_splits.__wrapped__() if hasattr(mock_splits, "__wrapped__") else mock_splits()
    test_stage1_no_train_test_overlap(splits)
    test_stage2_no_train_test_overlap(splits)
    print("✅ test_split.py — all tests passed")
