"""
page_analysis.py — SemVision "Pattern Analysis" page.

Features:
  - 8-panel wafer map grid (one example per defect class)
  - Per-class F1 bar chart (Plotly, traffic-light coloring)
  - Interactive confusion matrix (row-normalized percentage or raw-count views)
  - Most frequent directional error highlight with explanation
  - Worst performing class deep-dive
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from config_loader import STAGE2_CLASSES
from ui.reporting_copy import low_support_classes, low_support_warning
from visualization.wafer_viz import (
    WAFER_FAIL_COLOR,
    WAFER_MISSING_COLOR,
    WAFER_PASS_COLOR,
    wafer_to_rgb,
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.banner { background:linear-gradient(135deg,rgba(13,71,161,.4),rgba(25,118,210,.3));border-left:4px solid #4fc3f7;border-radius:8px;padding:.8rem 1.2rem;font-size:.85rem;color:#90caf9;margin-bottom:1.5rem; }
.section-header { font-size:1.2rem;font-weight:700;color:#fff;margin:1.8rem 0 .8rem;padding-bottom:.4rem;border-bottom:2px solid rgba(79,195,247,.3); }
.confusion-card { background:linear-gradient(135deg,rgba(183,28,28,.15),rgba(198,40,40,.1));border:1px solid rgba(239,83,80,.3);border-radius:14px;padding:1.4rem;margin:1rem 0; }
.confusion-title { font-size:1.1rem;font-weight:700;color:#ef5350;margin-bottom:.5rem; }
.explanation-box { background:rgba(79,195,247,.08);border-left:4px solid #4fc3f7;border-radius:8px;padding:.8rem 1.2rem;color:#90caf9;font-size:.9rem;margin:.5rem 0; }
.worst-class-box { background:rgba(255,213,79,.08);border-left:4px solid #ffd54f;border-radius:8px;padding:.8rem 1.2rem;color:#fff3cd;font-size:.9rem;margin:.5rem 0; }
</style>
"""

_CONFUSABLE_EXPLANATIONS = {
    ("Edge-Loc", "Edge-Ring"): "Both patterns appear at wafer edges. Edge-Loc has localized edge clusters; Edge-Ring forms a continuous ring at the edge — they differ mainly in angular extent and continuity.",
    ("Edge-Ring", "Edge-Loc"): "Both patterns appear at wafer edges. Edge-Ring is a near-complete ring; Edge-Loc covers only partial edge arcs — subtle at 64×64 resolution.",
    ("Center", "Donut"):       "Both involve the central wafer region. Donut has a characteristic hole in the center; Center shows uniform central failure — the hole can be lost at low resolution.",
    ("Donut", "Center"):       "Donut and Center both affect the central region. The distinguishing ring-shaped void of Donut can appear filled after nearest-neighbor downscaling.",
    ("Loc", "Random"):         "Local and Random patterns both show sparse, scattered fails. Loc has a spatially coherent cluster; Random is more uniformly distributed — difficult at 64×64.",
    ("Random", "Loc"):         "Random and Local patterns both appear as scattered failures. Distinguishing spatial coherence at low resolution is inherently ambiguous.",
    ("Near-Full", "Center"):   "Near-Full (nearly all dies fail) can resemble a very large Center cluster when the wafer boundary is mostly failure.",
    ("Scratch", "Random"):     "Thin Scratch patterns can degenerate to appear random after nearest-neighbor downsampling to 64×64.",
}


@st.cache_data
def _load_demo():
    p = os.path.join(_project_root, "data", "demo", "demo_samples.pkl")
    if not os.path.exists(p):
        return []
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data
def _load_stage2():
    p = os.path.join(_project_root, "artifacts", "reports", "stage2_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def show_analysis():
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="banner">ℹ️ <strong>Data:</strong> WM-811K semiconductor wafer maps (CC BY 4.0). '
        'This demo uses precomputed test-set samples. '
        'SemVision classifies wafer-level die patterns — not raw SEM/optical inspection images.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("# 🔬 Pattern Analysis")

    demo    = _load_demo()
    s2_met  = _load_stage2()

    if not demo or s2_met is None:
        st.warning("Training artifacts or demo samples not found. Run training pipeline first.")
        return

    # ── 1. Wafer map grid ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Example Wafer Maps by Defect Class</div>', unsafe_allow_html=True)
    st.markdown(
        "One representative test-set sample per defect class. Color: "
        f'<span style="color:{WAFER_PASS_COLOR}">●</span> Pass · '
        f'<span style="color:{WAFER_FAIL_COLOR}">●</span> Fail · '
        f'<span style="color:{WAFER_MISSING_COLOR}">●</span> Missing',
        unsafe_allow_html=True,
    )

    # Collect one sample per class
    class_samples = {}
    for samp in demo:
        cls = samp["true_class"]
        if cls in STAGE2_CLASSES and cls not in class_samples:
            class_samples[cls] = samp

    n_cols = 4
    rows   = [STAGE2_CLASSES[i:i+n_cols] for i in range(0, len(STAGE2_CLASSES), n_cols)]

    for row_cls in rows:
        cols = st.columns(n_cols)
        for col, cls in zip(cols, row_cls):
            with col:
                if cls in class_samples:
                    samp = class_samples[cls]
                    rgb  = wafer_to_rgb(samp["wafer_3ch"])
                    fig, ax = plt.subplots(figsize=(3, 3))
                    fig.patch.set_facecolor("#0e1117")
                    ax.set_facecolor("#0e1117")
                    ax.imshow(rgb, interpolation="nearest")
                    ax.set_title(cls, color="#4fc3f7", fontsize=9, fontweight="bold", pad=5)
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.markdown(f"**{cls}**\n\n*No sample*")

    # ── 2. Per-class F1 ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Per-Class F1 Score</div>', unsafe_allow_html=True)

    pcm         = s2_met["per_class_metrics"]
    cls_names   = [c for c in STAGE2_CLASSES if c in pcm]
    f1_scores   = [pcm[c]["f1"] for c in cls_names]

    from visualization.wafer_viz import plot_per_class_f1_plotly
    st.plotly_chart(plot_per_class_f1_plotly(cls_names, f1_scores), use_container_width=True)

    st.caption("🟢 F1 > 0.70 (Good)  ·  🟡 0.50 ≤ F1 ≤ 0.70 (Fair)  ·  🔴 F1 < 0.50 (Poor)")
    flagged_support = low_support_classes(pcm)
    if flagged_support:
        st.warning(
            "\n\n".join(
                low_support_warning(item["class_name"], item["support"])
                for item in flagged_support
            )
        )

    # ── 3. Confusion matrix ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)

    cm = np.array(s2_met["confusion_matrix"])
    control_col, download_col = st.columns([1.2, 1])
    with control_col:
        confusion_view = st.selectbox(
            "Confusion matrix view",
            ["Row-normalized %", "Raw counts"],
        )
    confusion_csv = pd.DataFrame(
        cm,
        index=pd.Index(STAGE2_CLASSES, name="True class"),
        columns=pd.Index(STAGE2_CLASSES, name="Predicted class"),
    ).to_csv()
    with download_col:
        st.download_button(
            "Download confusion matrix CSV",
            data=confusion_csv,
            file_name="semvision_stage2_confusion_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )
    from visualization.wafer_viz import plot_confusion_matrix_plotly
    st.plotly_chart(
        plot_confusion_matrix_plotly(cm, STAGE2_CLASSES, confusion_view),
        use_container_width=True,
    )

    # ── 4. Most frequent directional error ────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Most Frequent Directional Error</div>', unsafe_allow_html=True)

    wp = s2_met["worst_confusion_pair"]
    cls_a, cls_b, cnt = wp[0], wp[1], wp[2]

    explanation = _CONFUSABLE_EXPLANATIONS.get(
        (cls_a, cls_b),
        "These patterns share similar spatial characteristics at 64×64 resolution, "
        "making them inherently difficult to distinguish."
    )

    confusion_html = f"""
    <div class="confusion-card">
      <div class="confusion-title">True {cls_a} → Predicted {cls_b}: {cnt} test samples</div>
      <p style="color:#e0e0e0;margin:.3rem 0;">
        This is one direction of error; the reverse direction is not aggregated into this count.
      </p>
    </div>
    """
    st.markdown(confusion_html, unsafe_allow_html=True)

    st.markdown(
        f'<div class="explanation-box">📖 <strong>Why confused?</strong> {explanation}</div>',
        unsafe_allow_html=True,
    )

    # ── 5. Worst class ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Worst Performing Class</div>', unsafe_allow_html=True)

    wc      = s2_met["worst_class"]
    wc_met  = pcm.get(wc, {})
    wc_f1   = wc_met.get("f1", 0.0)
    wc_prec = wc_met.get("precision", 0.0)
    wc_rec  = wc_met.get("recall", 0.0)
    wc_sup  = wc_met.get("support", 0)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Worst Class",  wc)
    col_b.metric("F1 Score",     f"{wc_f1:.4f}")
    col_c.metric("Precision",    f"{wc_prec:.4f}")
    col_d.metric("Recall",       f"{wc_rec:.4f}")

    rec_msg = (
        "This class has very few training samples. "
        "Collecting more labeled examples or applying targeted oversampling may improve performance."
        if wc in ("Donut", "Near-Full")
        else "Consider inspecting misclassified samples to identify systematic confusion patterns."
    )
    st.markdown(
        f'<div class="worst-class-box">⚠️ <strong>Low-performance class: {wc}</strong> '
        f'(test support: {wc_sup} samples)<br>{rec_msg}</div>',
        unsafe_allow_html=True,
    )
