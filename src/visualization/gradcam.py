"""
gradcam.py — Grad-CAM visualization for Stage-2 CNN predictions.

compute_gradcam:
  - Finds last Conv2D layer automatically
  - Uses tf.GradientTape on class-specific output
  - Applies ReLU + normalize to [0,1]
  - Resizes the continuous explanation overlay to (64,64) with bilinear interpolation

visualize_gradcam_on_wafer:
  - 2-panel figure: Wafer map (fail channel) | Grad-CAM overlay
  - Styled with dark theme for premium look
  - Includes required caption about visual support vs root-cause
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import WAFER_SIZE
from visualization.wafer_viz import wafer_to_rgb


def resize_gradcam_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Bilinearly resize a continuous explanation map to wafer resolution."""
    resized = resize(
        np.asarray(heatmap, dtype=np.float32),
        (WAFER_SIZE, WAFER_SIZE),
        order=1,
        anti_aliasing=False,
        preserve_range=True,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def compute_gradcam(
    model: Any,
    input_array: np.ndarray,
    class_idx: int,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap.

    Parameters
    ----------
    model       : trained Keras model
    input_array : (1, 64, 64, 3) float32 array
    class_idx   : target class index (0-7)

    Returns
    -------
    heatmap : np.ndarray shape (64, 64), values in [0, 1]
    """
    # ── 1. Find last Conv2D layer ────────────────────────────────────────────
    import tensorflow as tf

    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")

    # ── 2. Gradient model ────────────────────────────────────────────────────
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [last_conv.output, model.output],
    )

    # ── 3. Gradient tape ─────────────────────────────────────────────────────
    input_tensor = tf.cast(input_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_output, predictions = grad_model(input_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)                    # (1, H, W, C)

    # ── 4. Pool gradients across spatial dims ────────────────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (C,)

    # ── 5. Weight feature maps by pooled grads ───────────────────────────────
    conv_out = conv_output[0]                                   # (H, W, C)
    weighted  = conv_out * pooled_grads                         # broadcast

    # ── 6. ReLU of weighted sum ──────────────────────────────────────────────
    heatmap = tf.reduce_mean(weighted, axis=-1)                 # (H, W)
    heatmap = tf.nn.relu(heatmap)

    # ── 7. Normalize to [0, 1] ───────────────────────────────────────────────
    hm_np = heatmap.numpy()
    if hm_np.max() - hm_np.min() > 1e-8:
        hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min())
    else:
        hm_np = np.zeros_like(hm_np)

    # Grad-CAM is continuous explanatory intensity, not categorical model
    # input. Bilinear interpolation is therefore appropriate for this overlay.
    return resize_gradcam_heatmap(hm_np)


def visualize_gradcam_on_wafer(
    wafer_3ch: np.ndarray,
    heatmap:   np.ndarray,
    predicted_class: str,
    true_class:      str,
    confidence:      float,
) -> plt.Figure:
    """
    Create a premium 2-panel Grad-CAM figure.

    Panel 1: categorical wafer die map using the shared palette
    Panel 2: continuous Grad-CAM activation over the categorical wafer map

    Parameters
    ----------
    wafer_3ch       : (64, 64, 3) float32 — channels [pass, fail, missing]
    heatmap         : (64, 64) float32 in [0, 1]
    predicted_class : string class name
    true_class      : string true class name
    confidence      : float [0, 1]

    Returns
    -------
    fig : matplotlib Figure
    """
    # Dark theme
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor":   "#1a1d23",
        "text.color":       "#ffffff",
        "axes.labelcolor":  "#ffffff",
        "xtick.color":      "#aaaaaa",
        "ytick.color":      "#aaaaaa",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#0e1117")

    wafer_rgb = wafer_to_rgb(wafer_3ch)

    # ── Panel 1: Wafer die map ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.imshow(wafer_rgb, interpolation="nearest")
    ax1.set_title("Wafer Die Map (Green=Pass, Red=Fail, Gray=Missing)",
                  color="#e0e0e0", fontsize=11, pad=8)
    ax1.axis("off")

    # ── Panel 2: Grad-CAM overlay ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.imshow(wafer_rgb, interpolation="nearest", alpha=0.75)
    im2 = ax2.imshow(heatmap, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    ax2.set_title("Grad-CAM Decision Regions",
                  color="#e0e0e0", fontsize=11, pad=8)
    ax2.axis("off")
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="#aaaaaa")
    cb2.set_label("Activation", color="#aaaaaa")

    # ── Main title ────────────────────────────────────────────────────────────
    correct = predicted_class == true_class
    title_color = "#4caf50" if correct else "#ef5350"
    fig.suptitle(
        f"Predicted: {predicted_class} ({confidence:.1%})  |  "
        f"Held-out Test Label: {true_class}",
        fontsize=14, fontweight="bold", color=title_color, y=1.01,
    )

    # ── Caption ───────────────────────────────────────────────────────────────
    caption = (
        "Grad-CAM shows regions influential to the model's prediction. "
        "This is visual decision support — not root-cause diagnosis. "
        "Engineering investigation is required to confirm process causes."
    )
    fig.text(
        0.5, -0.03, caption, ha="center", va="top",
        fontsize=9, color="#aaaaaa", style="italic",
        wrap=True,
    )

    plt.tight_layout(rect=[0, 0.0, 1, 1.0])

    # Reset rcParams so other plots aren't affected
    plt.rcParams.update(plt.rcParamsDefault)

    return fig
