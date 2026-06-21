"""
test_preprocess.py — Unit tests for resize_wafer_map.

Tests (no dataset needed):
  - Output shape is (64, 64, 3)
  - Channel values are strictly 0.0 or 1.0 (no fractional interpolation values)
  - Sum of 3 channels equals 1.0 at every pixel (exactly one die state per location)
"""
import os
import sys
import pytest
import numpy as np

# Add project root and src to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from data.preprocess import resize_wafer_map


def test_resize_shape():
    """Output shape must be (64, 64, 3)."""
    wafer = np.array([
        [0, 1, 2, 1, 0],
        [1, 2, 2, 2, 1],
        [2, 2, 1, 2, 2],
        [1, 2, 2, 2, 1],
        [0, 1, 2, 1, 0],
    ])
    result = resize_wafer_map(wafer)
    assert result.shape == (64, 64, 3), f"Expected (64,64,3), got {result.shape}"


def test_resize_binary_channels():
    """Channel values must be exactly 0.0 or 1.0 — no fractions from interpolation."""
    wafer = np.array([
        [0, 1, 2, 1, 0],
        [1, 2, 2, 2, 1],
        [2, 2, 1, 2, 2],
        [1, 2, 2, 2, 1],
        [0, 1, 2, 1, 0],
    ])
    result = resize_wafer_map(wafer)
    unique_vals = np.unique(result)
    for v in unique_vals:
        assert v in (0.0, 1.0), (
            f"Found fractional value {v:.6f} — channels must be binary. "
            "This indicates non-nearest-neighbor interpolation was used."
        )


def test_resize_channel_sum():
    """Sum of all 3 channels must equal exactly 1.0 at every pixel."""
    np.random.seed(42)
    # Generate a random wafer with valid categorical values
    wafer = np.random.choice([0, 1, 2], size=(28, 26))
    result = resize_wafer_map(wafer)
    channel_sum = result.sum(axis=-1)
    assert np.allclose(channel_sum, 1.0, atol=1e-6), (
        "Sum of 3 channels is not 1.0 at all pixels. "
        f"Min={channel_sum.min():.6f}, Max={channel_sum.max():.6f}"
    )


def test_resize_nonsquare_input():
    """Non-square inputs must also produce correct output."""
    wafer = np.zeros((10, 30), dtype=np.int8)
    wafer[2:8, 5:25] = 1
    wafer[3:7, 10:20] = 2
    result = resize_wafer_map(wafer)
    assert result.shape == (64, 64, 3)
    channel_sum = result.sum(axis=-1)
    assert np.allclose(channel_sum, 1.0, atol=1e-6)


if __name__ == "__main__":
    test_resize_shape()
    test_resize_binary_channels()
    test_resize_channel_sum()
    test_resize_nonsquare_input()
    print("✅ test_preprocess.py — all 4 tests passed")
