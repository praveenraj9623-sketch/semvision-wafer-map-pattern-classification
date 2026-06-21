import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from visualization.gradcam import resize_gradcam_heatmap, visualize_gradcam_on_wafer
from visualization.wafer_viz import (
    WAFER_FAIL_COLOR,
    WAFER_MISSING_COLOR,
    WAFER_PASS_COLOR,
    wafer_to_rgb,
)


def test_continuous_gradcam_resize_is_bilinear_and_bounded():
    source = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    resized = resize_gradcam_heatmap(source)
    assert resized.shape == (64, 64)
    assert np.isfinite(resized).all()
    assert resized.min() >= 0.0 and resized.max() <= 1.0
    assert np.any((resized > 0.0) & (resized < 1.0))


def test_categorical_wafer_uses_shared_exact_colors():
    wafer = np.zeros((1, 3, 3), dtype=np.float32)
    wafer[0, 0, 0] = 1.0
    wafer[0, 1, 1] = 1.0
    wafer[0, 2, 2] = 1.0
    rgb = wafer_to_rgb(wafer)
    np.testing.assert_allclose(rgb[0, 0], to_rgb(WAFER_PASS_COLOR))
    np.testing.assert_allclose(rgb[0, 1], to_rgb(WAFER_FAIL_COLOR))
    np.testing.assert_allclose(rgb[0, 2], to_rgb(WAFER_MISSING_COLOR))


def test_gradcam_figure_has_only_activation_colorbar():
    wafer = np.zeros((64, 64, 3), dtype=np.float32)
    wafer[:, :, 0] = 1.0
    wafer[16:32, 16:32, 0] = 0.0
    wafer[16:32, 16:32, 1] = 1.0
    heatmap = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    fig = visualize_gradcam_on_wafer(wafer, heatmap, "Center", "Center", 0.9)
    try:
        assert len(fig.axes) == 3  # two panels plus one activation colorbar
        assert [ax.get_ylabel() for ax in fig.axes].count("Activation") == 1
    finally:
        plt.close(fig)
