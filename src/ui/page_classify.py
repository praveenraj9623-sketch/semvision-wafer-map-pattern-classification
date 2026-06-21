"""
page_classify.py — SemVision "Classify a Wafer" page.

Features:
  - Sample selector dropdown
  - Wafer die map display (green/red/gray color coding)
  - Stage 1: No Labelled Failure Pattern vs Labelled Defect Pattern confidence gauge
  - Stage 2: 8-class softmax bar chart (Plotly) + correct/incorrect badge
  - Confidence abstention warning using the validation-selected threshold
  - Grad-CAM visualization with required caption
"""
import os
import sys
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

from config_loader import STAGE2_CLASSES, DEFAULT_CONFIDENCE_THRESHOLD
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
.banner { background: linear-gradient(135deg,rgba(13,71,161,.4),rgba(25,118,210,.3));border-left:4px solid #4fc3f7;border-radius:8px;padding:.8rem 1.2rem;font-size:.85rem;color:#90caf9;margin-bottom:1.5rem; }
.section-header { font-size:1.2rem;font-weight:700;color:#fff;margin:1.8rem 0 .8rem;padding-bottom:.4rem;border-bottom:2px solid rgba(79,195,247,.3); }
.result-card { background:linear-gradient(135deg,rgba(25,30,45,.95),rgba(18,22,35,.98));border:1px solid rgba(79,195,247,.2);border-radius:14px;padding:1.4rem;margin:.8rem 0; }
.correct-badge { background:linear-gradient(90deg,#1b5e20,#2e7d32);border:1px solid #43a047;border-radius:30px;padding:.5rem 1.4rem;font-weight:700;color:#a5d6a7;font-size:1rem;display:inline-block; }
.incorrect-badge { background:linear-gradient(90deg,#b71c1c,#c62828);border:1px solid #d32f2f;border-radius:30px;padding:.5rem 1.4rem;font-weight:700;color:#ffcdd2;font-size:1rem;display:inline-block; }
.uncertain-box { background:rgba(255,152,0,.12);border-left:4px solid #ff9800;border-radius:8px;padding:.8rem 1.2rem;color:#ffcc80;margin:.5rem 0; }
.stage-pipeline { display:flex;align-items:center;gap:.5rem;padding:.6rem 0; }
.stage-pill { background:rgba(79,195,247,.15);border:1px solid rgba(79,195,247,.3);border-radius:20px;padding:.3rem .9rem;font-size:.78rem;font-weight:600;color:#4fc3f7; }
.arrow { color:#555;font-size:1.2rem; }
.gate-decision-card { background:linear-gradient(135deg,rgba(20,25,40,.9),rgba(15,18,30,.95));border:1px solid rgba(79,195,247,.18);border-radius:12px;padding:.8rem 1rem;min-height:86px;display:flex;flex-direction:column;justify-content:center; }
.gate-decision-label { color:#9e9e9e;font-size:.78rem;font-weight:600;margin-bottom:.35rem; }
.gate-decision-value { font-size:clamp(1.05rem,1.55vw,1.45rem);line-height:1.2;font-weight:750;white-space:normal;overflow-wrap:normal;word-break:keep-all; }
.gate-decision-value.defect { color:#ef5350; }
.gate-decision-value.clear { color:#66bb6a; }
</style>
"""

FILTER_OPTIONS = [
    "All samples",
    "Correct predictions",
    "Incorrect predictions",
    "Uncertain predictions",
    "Per-class samples",
]


@st.cache_data
def _load_demo():
    demo_path = os.path.join(_project_root, "data", "demo", "demo_samples.pkl")
    if not os.path.exists(demo_path):
        return []
    with open(demo_path, "rb") as f:
        return pickle.load(f)


def _confidence_bar_plotly(
    class_names, confidences, predicted_idx, confidence_threshold
) -> go.Figure:
    """Build a descending Plotly probability chart with the prediction highlighted."""
    confidences = np.asarray(confidences, dtype=float)
    order = np.argsort(-confidences, kind="stable")
    sorted_names = [class_names[i] for i in order]
    sorted_confidences = confidences[order]
    predicted_class = class_names[predicted_idx]
    colors = []
    for class_name, confidence in zip(sorted_names, sorted_confidences):
        if class_name == predicted_class:
            colors.append("#4fc3f7")
        elif confidence > confidence_threshold * 0.7:
            colors.append("#455a64")
        else:
            colors.append("#263238")

    fig = go.Figure(go.Bar(
        y=sorted_names,
        x=sorted_confidences,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(79,195,247,0.4)", width=1),
        ),
        text=[f"{v:.2%}" for v in sorted_confidences],
        textposition="outside",
        textfont=dict(size=10, color="#9e9e9e"),
    ))
    fig.add_vline(x=confidence_threshold, line_dash="dot",
                  line_color="#ff9800", annotation_text=f"Threshold ({confidence_threshold:.0%})",
                  annotation_font_color="#ff9800", annotation_font_size=10)
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
        font=dict(color="#fff", family="Inter, sans-serif"),
        margin=dict(l=90, r=80, t=10, b=10),
        xaxis=dict(range=[0, 1.15], gridcolor="#2a2a2a", title="Confidence"),
        yaxis=dict(gridcolor="#2a2a2a", autorange="reversed"),
        height=320,
        showlegend=False,
    )
    return fig


def _sample_is_uncertain(sample: dict) -> bool:
    """Return whether an actually routed Stage 2 demo prediction abstains."""
    return bool(
        sample.get("stage1_prediction") == 1
        and sample.get("stage2_confidences") is not None
        and float(sample.get("stage2_confidence", 0.0))
        < float(sample.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    )


def _sample_is_correct(sample: dict) -> bool:
    """Return end-to-end correctness for a decisive demo prediction."""
    if _sample_is_uncertain(sample):
        return False
    if bool(sample.get("is_defect")):
        return bool(
            sample.get("stage1_prediction") == 1
            and sample.get("predicted_class") == sample.get("true_class")
        )
    return sample.get("stage1_prediction") == 0


def _filter_demo_samples(
    demo: list[dict], filter_mode: str, class_name: str | None = None
) -> list[dict]:
    """Filter immutable precomputed demo samples for the Classify selector."""
    if filter_mode == "All samples":
        return list(demo)
    if filter_mode == "Correct predictions":
        return [sample for sample in demo if _sample_is_correct(sample)]
    if filter_mode == "Incorrect predictions":
        return [
            sample
            for sample in demo
            if not _sample_is_uncertain(sample) and not _sample_is_correct(sample)
        ]
    if filter_mode == "Uncertain predictions":
        return [sample for sample in demo if _sample_is_uncertain(sample)]
    if filter_mode == "Per-class samples":
        return [sample for sample in demo if sample.get("true_class") == class_name]
    raise ValueError(f"Unsupported demo filter: {filter_mode}")


def _json_safe(value):
    """Convert demo payload values to JSON-safe structures without changing content."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _format_raw_failure_type(value) -> str:
    array = np.asarray(value, dtype=object)
    return str(array.reshape(-1)[0]) if array.size == 1 else str(value)


def show_classify():
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="banner">ℹ️ <strong>Data:</strong> WM-811K semiconductor wafer maps (CC BY 4.0). '
        'This demo uses precomputed test-set samples. '
        'SemVision classifies wafer-level die patterns — not raw SEM/optical inspection images.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("# 🔍 Classify a Wafer")
    st.markdown("Select a precomputed test-set sample to see the full two-stage classification pipeline.")

    demo = _load_demo()
    if not demo:
        st.warning("No demo samples found. Run `python src/data/prepare_demo.py` after training.")
        return

    # ── Pipeline visualization ────────────────────────────────────────────────
    st.markdown(
        '<div class="stage-pipeline">'
        '<span class="stage-pill">📥 Wafer Map</span>'
        '<span class="arrow">→</span>'
        '<span class="stage-pill">Stage 1: Binary Gate</span>'
        '<span class="arrow">→</span>'
        '<span class="stage-pill">Stage 2: Pattern CNN</span>'
        '<span class="arrow">→</span>'
        '<span class="stage-pill">Grad-CAM</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sample selector ───────────────────────────────────────────────────────
    filter_col, class_col = st.columns([1.2, 1])
    with filter_col:
        filter_mode = st.selectbox("Filter demo samples", FILTER_OPTIONS)
    class_name = None
    with class_col:
        if filter_mode == "Per-class samples":
            available_classes = [
                class_label
                for class_label in ["none", *STAGE2_CLASSES]
                if any(sample.get("true_class") == class_label for sample in demo)
            ]
            class_name = st.selectbox("Held-out test class", available_classes)
        else:
            st.caption("Choose “Per-class samples” to filter by held-out class.")

    filtered_demo = _filter_demo_samples(demo, filter_mode, class_name)
    if not filtered_demo:
        st.info("No precomputed demo samples match this filter.")
        return

    options = [
        f"Class: {s['true_class']}  —  Sample {s['sample_number']}"
        for s in filtered_demo
    ]
    choice   = st.selectbox("Select wafer sample", options)
    sample   = filtered_demo[options.index(choice)]
    confidence_threshold = float(
        sample.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    )

    st.markdown("---")

    # ── Wafer die map ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Wafer Die Map</div>', unsafe_allow_html=True)

    rgb = wafer_to_rgb(sample["wafer_3ch"])

    col_map, col_legend = st.columns([3, 1])
    with col_map:
        fig_map, ax_map = plt.subplots(figsize=(5, 5))
        fig_map.patch.set_facecolor("#0e1117")
        ax_map.set_facecolor("#0e1117")
        ax_map.imshow(rgb, interpolation="nearest")
        ax_map.set_title(f"Held-out Test Label: {sample['true_class']}",
                          color="#90caf9", fontsize=11, pad=8)
        ax_map.axis("off")
        st.pyplot(fig_map, use_container_width=False)
        plt.close(fig_map)
        st.caption(
            "This label is available because the demo uses precomputed held-out test samples."
        )

    with col_legend:
        st.markdown("**Color Legend**")
        st.markdown(
            f'<span style="color:{WAFER_PASS_COLOR};font-size:1.1rem;">●</span> Pass die',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="color:{WAFER_FAIL_COLOR};font-size:1.1rem;">●</span> Fail die',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="color:{WAFER_MISSING_COLOR};font-size:1.1rem;">●</span> Missing die',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(f"**Wafer size:** 64×64")
        pass_pct   = float(sample["wafer_3ch"][:,:,0].mean())
        fail_pct   = float(sample["wafer_3ch"][:,:,1].mean())
        missing_pct = float(sample["wafer_3ch"][:,:,2].mean())
        st.markdown(f"Pass: {pass_pct:.1%}")
        st.markdown(f"Fail: {fail_pct:.1%}")
        st.markdown(f"Missing: {missing_pct:.1%}")

    with st.expander("Technical details", expanded=False):
        original_shape = sample.get("original_shape", "unavailable")
        if isinstance(original_shape, (list, tuple, np.ndarray)):
            original_shape = " × ".join(str(dimension) for dimension in original_shape)
        technical_rows = [
            {
                "Field": "Source record ID",
                "Value": str(sample.get("source_record_id", "unavailable")),
            },
            {
                "Field": "Processed position",
                "Value": str(
                    sample.get("processed_position", sample.get("index", "unavailable"))
                ),
            },
            {"Field": "Original wafer shape", "Value": str(original_shape)},
            {
                "Field": "Held-out split",
                "Value": str(sample.get("split_name", "unavailable")),
            },
            {
                "Field": "Raw failure type",
                "Value": _format_raw_failure_type(sample.get("raw_failureType", "unavailable")),
            },
            {
                "Field": "Encoded Stage 2 target",
                "Value": str(sample.get("encoded_label", "unavailable")),
            },
        ]
        st.dataframe(
            pd.DataFrame(technical_rows), use_container_width=True, hide_index=True
        )
    with st.expander("View raw record payload", expanded=False):
        st.json(_json_safe(sample), expanded=False)

    st.markdown("---")

    # ── Stage 1 result ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Stage 1 — No Labelled Failure Pattern vs Labelled Defect Pattern</div>',
        unsafe_allow_html=True,
    )

    s1_pred  = sample["stage1_prediction"]
    s1_conf  = sample["stage1_confidence"]
    s1_label = "DEFECT PATTERN" if s1_pred == 1 else "NO LABELLED PATTERN"
    s1_class = "defect" if s1_pred == 1 else "clear"

    col1, col2, col3 = st.columns([1.2, 0.8, 1.25])
    with col1:
        st.markdown(
            '<div class="gate-decision-card">'
            '<div class="gate-decision-label">Gate Decision</div>'
            f'<div class="gate-decision-value {s1_class}">{s1_label}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    col2.metric("Confidence", f"{s1_conf:.1%}")
    col3.metric(
        "Held-out Test Label",
        "Labelled Defect Pattern" if sample["is_defect"] else "No Labelled Failure Pattern",
    )
    st.caption(
        "Technical labels: “Labelled Defect Pattern” / “No Labelled Failure Pattern”. "
        f"Validation-selected gate threshold: "
        f"{sample.get('stage1_decision_threshold', 0.5):.2f}"
    )

    # Confidence progress bar
    st.progress(s1_conf, text=f"Stage 1 confidence: {s1_conf:.1%}")

    # ── Stage 2 result (only when the Stage 1 gate predicts defect) ───────────
    if s1_pred == 1 and sample["stage2_confidences"] is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">Stage 2 — Defect Pattern Classification</div>', unsafe_allow_html=True)

        pred_cls  = sample["predicted_class"]
        true_cls  = sample["true_class"]
        s2_conf   = sample["stage2_confidence"]
        s2_confs  = sample["stage2_confidences"]
        pred_idx  = STAGE2_CLASSES.index(pred_cls) if pred_cls in STAGE2_CLASSES else 0

        # Confidence chart
        fig_conf = _confidence_bar_plotly(
            STAGE2_CLASSES, s2_confs, pred_idx, confidence_threshold
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        st.caption(
            "Probability bars are sorted by model confidence; the threshold applies to the "
            "maximum predicted probability."
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Class", pred_cls)
        col_b.metric("Held-out Test Label", true_cls)
        col_c.metric("Confidence",      f"{s2_conf:.1%}")

        # Correct/incorrect badge
        is_correct = (pred_cls == true_cls)
        if is_correct:
            st.markdown(
                '<div style="text-align:center;margin:.6rem 0;">'
                '<span class="correct-badge">✅ Correct Prediction</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center;margin:.6rem 0;">'
                f'<span class="incorrect-badge">❌ Incorrect — Held-out Test Label: {true_cls}</span></div>',
                unsafe_allow_html=True,
            )

        # Confidence threshold warning
        if s2_conf < confidence_threshold:
            st.markdown(
                f'<div class="uncertain-box">⚠️ <strong>Model confidence too low — result uncertain.</strong><br>'
                f'Max softmax {s2_conf:.1%} &lt; threshold {confidence_threshold:.0%}. '
                f'Consider treating as UNCERTAIN and escalating to engineering review.</div>',
                unsafe_allow_html=True,
            )

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🌡️ Grad-CAM — Decision Regions</div>', unsafe_allow_html=True)

        if sample["gradcam_heatmap"] is not None:
            from visualization.gradcam import visualize_gradcam_on_wafer
            fig_gcam = visualize_gradcam_on_wafer(
                sample["wafer_3ch"],
                sample["gradcam_heatmap"],
                pred_cls, true_cls, s2_conf,
            )
            st.pyplot(fig_gcam, use_container_width=True)
            plt.close(fig_gcam)
        else:
            st.info("Grad-CAM not available for this sample.")

    elif s1_pred == 0:
        st.info(
            "✅ Stage 1 classified this wafer as having no labelled failure pattern — "
            "Stage 2 was not invoked."
        )
