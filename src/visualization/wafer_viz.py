"""
wafer_viz.py — Production-quality visualization functions for SemVision.

All functions return matplotlib Figures styled with a dark, premium aesthetic.
Plotly versions are also provided (_plotly suffix) for interactive Streamlit use.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import plotly.graph_objects as go
import plotly.express as px

# ─── Shared theme ─────────────────────────────────────────────────────────────
_BG_DARK      = "#0e1117"
_BG_CARD      = "#1a1d23"
_ACCENT_BLUE  = "#4fc3f7"
_ACCENT_GREEN = "#66bb6a"
_ACCENT_RED   = "#ef5350"
_ACCENT_GOLD  = "#ffd54f"
_TEXT_PRIMARY = "#ffffff"
_TEXT_MUTED   = "#9e9e9e"

# One categorical palette for every wafer rendering and its legends.
WAFER_PASS_COLOR = "#33B833"
WAFER_FAIL_COLOR = "#F03333"
WAFER_MISSING_COLOR = "#595959"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=_BG_DARK,
    plot_bgcolor=_BG_CARD,
    font=dict(color=_TEXT_PRIMARY, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=60, b=40),
)


def _dark_fig(figsize=(8, 6)) -> tuple:
    """Return (fig, ax) with dark background styling."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_CARD)
    ax.tick_params(colors=_TEXT_MUTED)
    ax.spines[:].set_color("#3a3a3a")
    ax.xaxis.label.set_color(_TEXT_MUTED)
    ax.yaxis.label.set_color(_TEXT_MUTED)
    ax.title.set_color(_TEXT_PRIMARY)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Wafer map
# ─────────────────────────────────────────────────────────────────────────────

def wafer_to_rgb(wafer_3ch: np.ndarray) -> np.ndarray:
    """Convert categorical [pass, fail, missing] channels to the shared RGB palette."""
    wafer = np.asarray(wafer_3ch)
    if wafer.ndim != 3 or wafer.shape[-1] != 3:
        raise ValueError("wafer_3ch must have shape (height, width, 3)")

    rgb = np.zeros((*wafer.shape[:2], 3), dtype=np.float32)
    rgb[wafer[:, :, 0] == 1.0] = to_rgb(WAFER_PASS_COLOR)
    rgb[wafer[:, :, 1] == 1.0] = to_rgb(WAFER_FAIL_COLOR)
    rgb[wafer[:, :, 2] == 1.0] = to_rgb(WAFER_MISSING_COLOR)
    return rgb


def plot_wafer_map(wafer_3ch: np.ndarray, title: str, ax=None) -> plt.Axes:
    """
    Show wafer map with explicit color coding:
      Green  → pass die  (channel 0)
      Red    → fail die  (channel 1)
      Gray   → missing die (channel 2)

    Parameters
    ----------
    wafer_3ch : (64, 64, 3) float32
    title     : plot title string
    ax        : existing matplotlib Axes, or None to create new

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(_BG_DARK)
        ax.set_facecolor(_BG_CARD)

    ax.imshow(wafer_to_rgb(wafer_3ch), interpolation="nearest")
    ax.set_title(title, color=_TEXT_PRIMARY, fontsize=10, pad=6)
    ax.axis("off")

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Class distribution — Plotly (interactive)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution_plotly(
    class_counts_dict: dict,
    title: str = "Defect Class Distribution",
) -> go.Figure:
    """Interactive horizontal bar chart of class distribution (Plotly)."""
    sorted_items = sorted(class_counts_dict.items(), key=lambda x: x[1])
    labels  = [k for k, _ in sorted_items]
    values  = [v for _, v in sorted_items]
    total   = sum(values)
    pcts    = [v / total * 100 for v in values]
    colors  = [_ACCENT_RED if "Donut" in l or "Near" in l else _ACCENT_BLUE for l in labels]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v:,} ({p:.1f}%)" for v, p in zip(values, pcts)],
            textposition="outside",
            textfont=dict(color=_TEXT_MUTED, size=11),
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=16, color=_TEXT_PRIMARY)),
        xaxis=dict(title="Sample Count", gridcolor="#2a2a2a"),
        yaxis=dict(title="", gridcolor="#2a2a2a"),
        height=420,
    )
    return fig


def plot_class_distribution(class_counts_dict: dict) -> plt.Figure:
    """Matplotlib version of class distribution for fallback."""
    sorted_items = sorted(class_counts_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = _dark_fig(figsize=(9, 5))
    colors  = [_ACCENT_BLUE] * len(labels)
    bars    = ax.barh(labels, values, color=colors, height=0.6, edgecolor="none")
    ax.set_xlabel("Count", color=_TEXT_MUTED)
    ax.set_title("Defect Class Distribution", color=_TEXT_PRIMARY, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}", va="center", color=_TEXT_MUTED, fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix — Plotly (interactive)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_plotly(
    cm: np.ndarray,
    class_names: list,
    view_mode: str = "Row-normalized %",
) -> go.Figure:
    """Interactive confusion matrix in row-normalized or raw-count view."""
    if view_mode not in {"Row-normalized %", "Raw counts"}:
        raise ValueError(f"Unsupported confusion-matrix view: {view_mode}")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))
    normalized = view_mode == "Row-normalized %"
    display_values = cm_norm if normalized else cm
    color_threshold = 0.45 if normalized else float(cm.max()) * 0.45

    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(dict(
                x=class_names[j], y=class_names[i],
                text=f"{cm_norm[i,j]:.1%}" if normalized else f"{cm[i,j]:,}",
                showarrow=False,
                font=dict(
                    color="white" if display_values[i, j] > color_threshold else _TEXT_MUTED,
                    size=11,
                ),
            ))

    heatmap_kwargs = {
        "z": display_values,
        "x": class_names,
        "y": class_names,
        "customdata": cm,
        "colorscale": "Blues",
        "showscale": True,
        "colorbar": {"title": "% of true class" if normalized else "Test samples"},
        "hovertemplate": (
            "True: %{y}<br>Predicted: %{x}<br>Row share: %{z:.1%}"
            "<br>Test samples: %{customdata}<extra></extra>"
            if normalized
            else "True: %{y}<br>Predicted: %{x}<br>Test samples: %{z}<extra></extra>"
        ),
    }
    if normalized:
        heatmap_kwargs.update({"zmin": 0, "zmax": 1})

    fig = go.Figure(
        go.Heatmap(**heatmap_kwargs)
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"Confusion Matrix — {view_mode}", font=dict(size=16, color=_TEXT_PRIMARY)),
        xaxis=dict(title="Predicted", side="bottom", tickangle=-30, automargin=True),
        yaxis=dict(title="True", autorange="reversed", automargin=True),
        annotations=annotations,
        height=570,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> plt.Figure:
    """Matplotlib annotated heatmap confusion matrix."""
    fig, ax = _dark_fig(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else _TEXT_MUTED, fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history_plotly(history_dict: dict) -> go.Figure:
    """Interactive training history with dual-axis (Plotly)."""
    epochs = list(range(1, len(history_dict["loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["loss"],
                             name="Train Loss",     line=dict(color=_ACCENT_RED),   mode="lines+markers"))
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["val_loss"],
                             name="Val Loss",       line=dict(color=_ACCENT_RED, dash="dash"), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["accuracy"],
                             name="Train Acc",      line=dict(color=_ACCENT_BLUE),  mode="lines+markers",
                             yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["val_accuracy"],
                             name="Val Acc",        line=dict(color=_ACCENT_BLUE, dash="dash"), mode="lines+markers",
                             yaxis="y2"))
    best_epoch = history_dict.get("best_epoch")
    if best_epoch is not None:
        fig.add_vline(
            x=best_epoch,
            line_dash="dot",
            line_color=_ACCENT_GOLD,
            annotation_text=f"Restored epoch {best_epoch}",
            annotation_font_color=_ACCENT_GOLD,
        )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Training History", font=dict(size=16, color=_TEXT_PRIMARY)),
        xaxis=dict(title="Epoch", gridcolor="#2a2a2a"),
        yaxis=dict(title="Loss", gridcolor="#2a2a2a"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        height=420,
    )
    return fig


def plot_training_history(history_dict: dict) -> plt.Figure:
    """Matplotlib dual-axis training history."""
    fig, ax1 = _dark_fig(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax2.set_facecolor(_BG_CARD)
    ax2.tick_params(colors=_TEXT_MUTED)

    epochs = range(1, len(history_dict["loss"]) + 1)

    ax1.plot(epochs, history_dict["loss"],         color=_ACCENT_RED,   label="Train Loss")
    ax1.plot(epochs, history_dict["val_loss"],      color=_ACCENT_RED,   linestyle="--", label="Val Loss")
    ax1.set_ylabel("Loss", color=_TEXT_MUTED)
    ax1.set_xlabel("Epoch", color=_TEXT_MUTED)

    ax2.plot(epochs, history_dict["accuracy"],      color=_ACCENT_BLUE,  label="Train Acc")
    ax2.plot(epochs, history_dict["val_accuracy"],  color=_ACCENT_BLUE,  linestyle="--", label="Val Acc")
    ax2.set_ylabel("Accuracy", color=_TEXT_MUTED)
    best_epoch = history_dict.get("best_epoch")
    if best_epoch is not None:
        ax1.axvline(
            best_epoch,
            color=_ACCENT_GOLD,
            linestyle=":",
            linewidth=1.5,
            label=f"Restored epoch {best_epoch}",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=_BG_CARD, labelcolor=_TEXT_MUTED, loc="upper right")
    ax1.set_title("Training History", color=_TEXT_PRIMARY, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-class F1 — Plotly (interactive)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1_plotly(class_names: list, f1_scores: list) -> go.Figure:
    """Interactive horizontal bar chart with traffic-light coloring (Plotly)."""
    colors = [
        _ACCENT_GREEN if f > 0.7 else (_ACCENT_GOLD if f >= 0.5 else _ACCENT_RED)
        for f in f1_scores
    ]
    sorted_pairs = sorted(zip(class_names, f1_scores, colors), key=lambda x: x[1])
    snames, sf1, scolors = zip(*sorted_pairs)

    fig = go.Figure(
        go.Bar(
            y=list(snames),
            x=list(sf1),
            orientation="h",
            marker_color=list(scolors),
            text=[f"{v:.3f}" for v in sf1],
            textposition="outside",
            textfont=dict(color=_TEXT_MUTED, size=11),
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Per-Class F1 Score", font=dict(size=16, color=_TEXT_PRIMARY)),
        xaxis=dict(title="F1 Score", range=[0, 1.15], gridcolor="#2a2a2a"),
        yaxis=dict(title=""),
        height=400,
    )
    return fig


def plot_per_class_f1(class_names: list, f1_scores: list) -> plt.Figure:
    """Matplotlib per-class F1 horizontal bar chart with traffic-light coloring."""
    fig, ax = _dark_fig(figsize=(9, 5))
    colors = [
        _ACCENT_GREEN if f > 0.7 else (_ACCENT_GOLD if f >= 0.5 else _ACCENT_RED)
        for f in f1_scores
    ]
    bars = ax.barh(class_names, f1_scores, color=colors, height=0.6, edgecolor="none")
    ax.set_xlabel("F1 Score", color=_TEXT_MUTED)
    ax.set_xlim(0, 1.15)
    ax.set_title("Per-Class F1 Score", color=_TEXT_PRIMARY, fontweight="bold")
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", color=_TEXT_MUTED, fontsize=9)
    legend_patches = [
        mpatches.Patch(facecolor=_ACCENT_GREEN, label="F1 > 0.70"),
        mpatches.Patch(facecolor=_ACCENT_GOLD,  label="0.50 ≤ F1 ≤ 0.70"),
        mpatches.Patch(facecolor=_ACCENT_RED,   label="F1 < 0.50"),
    ]
    ax.legend(handles=legend_patches, facecolor=_BG_CARD,
              labelcolor=_TEXT_MUTED, loc="lower right")
    plt.tight_layout()
    return fig
