"""
page_evaluation.py — SemVision "Model Evaluation" page.

Features:
  - Full classification report table (sortable)
  - Interactive training history plots (Plotly)
  - Stage 1 vs Stage 2 summary card
  - Model card viewer (expandable, section-labeled)
  - Calibration insight (correct vs incorrect softmax confidence)
"""
import os
import sys
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ui.reporting_copy import (
    AUGMENTATION_DESCRIPTION,
    COHORT_TRANSPARENCY_STATEMENT,
    STAGE2_CONDITIONAL_NOTE,
    cohort_summary_html,
    cohort_summary_rows,
    low_support_badge,
    low_support_classes,
    low_support_warning,
)

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.banner { background:linear-gradient(135deg,rgba(13,71,161,.4),rgba(25,118,210,.3));border-left:4px solid #4fc3f7;border-radius:8px;padding:.8rem 1.2rem;font-size:.85rem;color:#90caf9;margin-bottom:1.5rem; }
.section-header { font-size:1.2rem;font-weight:700;color:#fff;margin:1.8rem 0 .8rem;padding-bottom:.4rem;border-bottom:2px solid rgba(79,195,247,.3); }
.summary-card { background:linear-gradient(135deg,rgba(25,30,45,.95),rgba(18,22,35,.98));border:1px solid rgba(79,195,247,.2);border-radius:14px;padding:1.4rem;margin:.8rem 0; }
.mc-section { background:rgba(79,195,247,.05);border-left:3px solid rgba(79,195,247,.4);border-radius:6px;padding:.7rem 1rem;margin:.5rem 0; }
.mc-label { font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#4fc3f7;margin-bottom:.25rem; }
.calibration-bar { display:flex;gap:1rem;align-items:center;margin:.5rem 0; }
.cal-label { font-size:.85rem;color:#9e9e9e;width:180px; }
.cal-fill-correct   { background:linear-gradient(90deg,#66bb6a,#81c784);height:22px;border-radius:4px;min-width:4px; }
.cal-fill-incorrect { background:linear-gradient(90deg,#ef5350,#e57373);height:22px;border-radius:4px;min-width:4px; }
.cal-value { font-size:.85rem;font-weight:600;color:#e0e0e0;margin-left:.5rem; }
.cohort-note { background:rgba(79,195,247,.06);border:1px solid rgba(79,195,247,.18);border-radius:10px;padding:.9rem 1rem;color:#cbd5e1;font-size:.88rem;line-height:1.55; }
</style>
"""

def _build_end_to_end_display(e2e: dict) -> tuple[list[dict], list[dict]]:
    """Build mutually exclusive final outcomes plus a separate uncertainty subset."""
    counts = e2e.get("counts", {})
    uncertain_total = int(counts.get("uncertain_predictions", 0))
    uncertain_defect = int(counts.get("uncertain_true_defect", 0))
    uncertain_none = int(counts.get("uncertain_true_none", 0))
    decisive_none_to_stage2 = (
        int(counts.get("no_labelled_pattern_sent_to_stage2", 0)) - uncertain_none
    )

    outcomes = [
        {
            "Final routing outcome": "Correctly identified no-labelled-pattern",
            "Count": int(counts.get("correctly_identified_no_labelled_pattern", 0)),
        },
        {
            "Final routing outcome": (
                "Stage 1 false positive → Stage 2 decisive prediction"
            ),
            "Count": decisive_none_to_stage2,
        },
        {
            "Final routing outcome": "Stage 1 false negative → defect-pattern missed",
            "Count": int(counts.get("defect_pattern_missed_by_stage1", 0)),
        },
        {
            "Final routing outcome": "Correctly classified defect-pattern",
            "Count": int(counts.get("correctly_classified_defect_pattern", 0)),
        },
        {
            "Final routing outcome": (
                "Defect-pattern routed to Stage 2 but misclassified"
            ),
            "Count": int(
                counts.get("defect_pattern_sent_to_stage2_but_misclassified", 0)
            ),
        },
        {"Final routing outcome": "Uncertain final prediction", "Count": uncertain_total},
    ]
    uncertainty = [
        {"Uncertainty subset": "Total uncertain", "Count": uncertain_total},
        {
            "Uncertainty subset": "Uncertain true defect-pattern",
            "Count": uncertain_defect,
        },
        {
            "Uncertainty subset": "Uncertain true no-labelled-pattern",
            "Count": uncertain_none,
        },
    ]

    if any(row["Count"] < 0 for row in outcomes):
        raise ValueError("End-to-end source counts cannot produce valid final outcomes")
    if uncertain_defect + uncertain_none != uncertain_total:
        raise ValueError("Uncertainty breakdown does not equal total uncertainty")
    if sum(row["Count"] for row in outcomes) != int(e2e["global_test_samples"]):
        raise ValueError("Final routing outcomes do not equal the global test count")
    return outcomes, uncertainty


@st.cache_data
def _load_all():
    reports_dir = os.path.join(_project_root, "artifacts", "reports")
    data = {}
    for key, fname in [
        ("s1", "stage1_metrics.json"),
        ("s2", "stage2_metrics.json"),
        ("mc", "model_card.json"),
        ("hist", "training_history.json"),
        ("e2e", "end_to_end_metrics.json"),
        ("dataset", "dataset_summary.json"),
    ]:
        p = os.path.join(reports_dir, fname)
        if os.path.exists(p):
            with open(p) as f:
                data[key] = json.load(f)
        else:
            data[key] = None
    return data


def show_evaluation():
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="banner">ℹ️ <strong>Data:</strong> WM-811K semiconductor wafer maps (CC BY 4.0). '
        'This demo uses precomputed test-set samples. '
        'SemVision classifies wafer-level die patterns — not raw SEM/optical inspection images.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("# 📋 Model Evaluation")

    d = _load_all()

    if not d["s2"]:
        st.warning("Training artifacts not found. Run `python run_training.py` first.")
        return

    s1   = d["s1"]
    s2   = d["s2"]
    mc   = d["mc"]
    hist = d["hist"]
    e2e  = d["e2e"]
    dataset = d["dataset"]

    from config_loader import STAGE2_CLASSES, DEFAULT_CONFIDENCE_THRESHOLD

    if dataset:
        st.markdown('<div class="section-header">🧾 Dataset Cohort & Provenance</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="cohort-note">{COHORT_TRANSPARENCY_STATEMENT}</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(cohort_summary_rows(dataset)),
            use_container_width=True,
            hide_index=True,
        )

    # ── 1. Classification report ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Stage 2 Classification Report</div>', unsafe_allow_html=True)

    pcm = s2["per_class_metrics"]
    rows = []
    for cls in STAGE2_CLASSES:
        if cls in pcm:
            m = pcm[cls]
            rows.append({
                "Class":     cls,
                "Precision": m["precision"],
                "Recall":    m["recall"],
                "F1 Score":  m["f1"],
                "Support":   m["support"],
                "Support status": low_support_badge(m["support"]),
            })

    rows.append({"Class": "— Overall Macro",    "Precision": "—", "Recall": "—",
                 "F1 Score": s2["macro_f1"],    "Support": "—", "Support status": ""})
    rows.append({"Class": "— Overall Weighted", "Precision": "—", "Recall": "—",
                 "F1 Score": s2["weighted_f1"], "Support": "—", "Support status": ""})

    df_report = pd.DataFrame(rows)

    # Apply visual formatting
    def _color_f1(val):
        if not isinstance(val, float):
            return ""
        if val >= 0.70:
            return "background-color: rgba(102,187,106,0.15); color: #a5d6a7;"
        if val >= 0.50:
            return "background-color: rgba(255,213,79,0.12); color: #fff176;"
        return "background-color: rgba(239,83,80,0.15); color: #ffcdd2;"

    def _color_support(val):
        if isinstance(val, str) and val.startswith("Low support"):
            return (
                "background-color: rgba(255,193,7,0.18); color: #ffe082; "
                "font-weight: 700;"
            )
        return ""

    styled = (
        df_report.style
        .map(_color_f1, subset=["F1 Score"])
        .map(_color_support, subset=["Support status"])
    )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Support status": st.column_config.TextColumn(
                "Support status",
                width="medium",
                help="Metrics with fewer than 50 held-out test samples.",
            )
        },
    )
    flagged_support = low_support_classes(pcm)
    if flagged_support:
        st.warning(
            "\n\n".join(
                low_support_warning(item["class_name"], item["support"])
                for item in flagged_support
            )
        )

    # Accuracy / balanced accuracy
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Overall Accuracy",       f"{s2['accuracy']:.4f}")
    col_b.metric("Balanced Accuracy",      f"{s2['balanced_accuracy']:.4f}")
    col_c.metric("Inference Latency",      f"{s2['inference_latency_ms']:.1f} ms")

    # ── 2. Training history ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Training History</div>', unsafe_allow_html=True)

    if hist:
        from visualization.wafer_viz import plot_training_history_plotly
        st.plotly_chart(plot_training_history_plotly(hist), use_container_width=True)
        if hist.get("best_epoch"):
            st.caption(
                f"The deployed model restored the best {hist.get('checkpoint_metric', 'validation')} "
                f"weights from epoch {hist['best_epoch']}; epoch {hist.get('epochs_ran')} "
                "was the final training epoch, not the deployed checkpoint."
            )
    else:
        st.info("Training history not available. It is saved automatically during `run_training.py`.")

    # ── 3. Confidence separation (not calibration) ───────────────────────────
    st.markdown('<div class="section-header">🎯 Confidence Separation Analysis</div>', unsafe_allow_html=True)
    st.caption(
        "Mean confidence on correct versus incorrect predictions measures "
        "separation; it is not, by itself, a calibration metric."
    )

    cal_correct   = s2.get("max_softmax_correct",   0.0)
    cal_incorrect = s2.get("max_softmax_incorrect",  0.0)

    cal_html = f"""
    <div class="summary-card">
      <div class="calibration-bar">
        <span class="cal-label">Correct predictions</span>
        <div class="cal-fill-correct" style="width:{cal_correct*260:.0f}px"></div>
        <span class="cal-value">{cal_correct:.3f}</span>
      </div>
      <div class="calibration-bar">
        <span class="cal-label">Incorrect predictions</span>
        <div class="cal-fill-incorrect" style="width:{cal_incorrect*260:.0f}px"></div>
        <span class="cal-value">{cal_incorrect:.3f}</span>
      </div>
      <p style="color:#9e9e9e;font-size:.82rem;margin-top:.7rem;">
        Validation-selected abstention threshold: {s2.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD):.0%}
        — predictions below this are marked UNCERTAIN.
      </p>
    </div>
    """
    st.markdown(cal_html, unsafe_allow_html=True)

    calibration = s2.get("calibration", {})
    if calibration:
        st.markdown('<div class="section-header">📐 Calibration Quality</div>', unsafe_allow_html=True)
        cal_a, cal_b = st.columns(2)
        cal_a.metric(
            "Expected Calibration Error",
            f"{calibration.get('expected_calibration_error', 0.0):.4f}",
        )
        cal_b.metric(
            "Multiclass Brier Score",
            f"{calibration.get('multiclass_brier_score', 0.0):.4f}",
        )
        populated = [
            item for item in calibration.get("reliability_bins", [])
            if item.get("count", 0) > 0
        ]
        if populated:
            reliability = go.Figure()
            reliability.add_trace(go.Scatter(
                x=[item["average_confidence"] for item in populated],
                y=[item["accuracy"] for item in populated],
                mode="lines+markers",
                name="Observed accuracy",
                marker=dict(size=8, color="#4fc3f7"),
            ))
            reliability.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                line=dict(color="#9e9e9e", dash="dash"),
            ))
            reliability.update_layout(
                title="Reliability Diagram",
                xaxis=dict(title="Mean confidence", range=[0, 1]),
                yaxis=dict(title="Observed accuracy", range=[0, 1]),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1d23",
                font=dict(color="#ffffff"),
                height=380,
            )
            st.plotly_chart(reliability, use_container_width=True)
            confidence_bins = calibration.get("reliability_bins", [])
            bin_counts = go.Figure(go.Bar(
                x=[
                    f"{float(item['lower']):.1f}–{float(item['upper']):.1f}"
                    for item in confidence_bins
                ],
                y=[int(item.get("count", 0)) for item in confidence_bins],
                marker_color="#4fc3f7",
                text=[f"{int(item.get('count', 0)):,}" for item in confidence_bins],
                textposition="outside",
            ))
            bin_counts.update_layout(
                title="Confidence-Bin Sample Counts",
                xaxis=dict(title="Maximum predicted probability bin"),
                yaxis=dict(title="Test samples", gridcolor="#2a2a2a"),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1d23",
                font=dict(color="#ffffff"),
                margin=dict(l=55, r=20, t=55, b=55),
                height=260,
                showlegend=False,
            )
            st.plotly_chart(bin_counts, use_container_width=True)
            st.caption(
                "Per-bin sample counts provide support context for the reliability diagram; "
                "they do not change the calibration calculations."
            )

    # ── 4. End-to-end pipeline ──────────────────────────────────────────────
    if e2e:
        st.markdown('<div class="section-header">🔗 End-to-End Pipeline</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        e1.metric("Nine-Class Macro F1", f"{e2e['end_to_end_macro_f1']:.4f}")
        e2.metric("Defect Gate Recall", f"{e2e['end_to_end_defect_recall']:.4f}")
        e3.metric("Abstention Rate", f"{e2e['abstention_rate']:.2%}")
        outcomes, uncertainty = _build_end_to_end_display(e2e)
        st.dataframe(pd.DataFrame(outcomes), use_container_width=True, hide_index=True)
        st.markdown("**Uncertainty breakdown (subset)**")
        st.dataframe(pd.DataFrame(uncertainty), use_container_width=True, hide_index=True)
        st.caption(
            "Uncertainty overlays Stage 2 predictions in the source report. The final routing "
            "outcomes above are mutually exclusive; the uncertainty breakdown is a subset and "
            "should not be summed again with those outcomes."
        )

    # ── 5. Stage 1 vs Stage 2 summary ────────────────────────────────────────
    st.markdown('<div class="section-header">🏁 Pipeline Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    if s1:
        col1.metric("Stage 1 Champion",  s1["champion"].replace("_", " ").title())
        col2.metric("Stage 1 ROC-AUC",   f"{s1['champion_metrics']['roc_auc']:.4f}")
    col3.metric("Stage 2 Conditional Macro F1", f"{s2['macro_f1']:.4f}")
    col4.metric("Stage 2 Worst-Class F1",
                f"{s2['per_class_metrics'][s2['worst_class']]['f1']:.4f} ({s2['worst_class']})")

    # ── 6. Model card viewer ──────────────────────────────────────────────────
    if mc:
        st.markdown('<div class="section-header">📄 Model Card</div>', unsafe_allow_html=True)

        with st.expander("📋 View Full Model Card", expanded=False):
            augmentation_display = (
                AUGMENTATION_DESCRIPTION
                if mc.get("augmentation_used") == "with_augmentation"
                else mc.get("augmentation_used", "")
            )
            sections = [
                ("Model Name",           mc.get("model_name", "")),
                ("Model Type",           mc.get("model_type", "")),
                ("Domain Note",          mc.get("domain_note", "")),
                ("Intended Use",         mc.get("intended_use", "")),
                ("Out of Scope",         mc.get("out_of_scope", "")),
                ("Training Data",        mc.get("training_data", "")),
                ("Data Split Method",    mc.get("data_split_method", "")),
                ("Augmentation Used",    augmentation_display),
                ("Random Seed",          str(mc.get("random_seed", ""))),
                ("Inference Latency",    f"{mc.get('inference_latency_ms', 0):.1f} ms"),
                ("Worst Performing Class", mc.get("worst_class", "")),
            ]
            if dataset:
                sections.extend([
                    (
                        "Cohort Transparency",
                        f"{COHORT_TRANSPARENCY_STATEMENT}<br><br>"
                        f"{cohort_summary_html(dataset)}",
                    ),
                    ("Stage 2 Metric Scope", STAGE2_CONDITIONAL_NOTE),
                ])

            for label, text in sections:
                st.markdown(
                    f'<div class="mc-section"><div class="mc-label">{label}</div>'
                    f'<div style="color:#e0e0e0;font-size:.9rem;">{text}</div></div>',
                    unsafe_allow_html=True,
                )

            # Performance table
            st.markdown('<div class="mc-section"><div class="mc-label">Performance</div>', unsafe_allow_html=True)
            perf = {
                key: value
                for key, value in mc.get("performance", {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            explicit_performance = [
                {"Metric": "Stage 2 Conditional Macro F1", "Value": f"{s2['macro_f1']:.4f}"},
                {
                    "Metric": "End-to-End Nine-Class Macro F1",
                    "Value": f"{e2e['end_to_end_macro_f1']:.4f}" if e2e else "unavailable",
                },
                {
                    "Metric": "Stage 1 Defect Gate Recall",
                    "Value": f"{s1['champion_metrics']['defect_pattern_recall']:.4f}" if s1 else "unavailable",
                },
                {
                    "Metric": "Stage 1 validation-selected threshold",
                    "Value": f"{float(s1.get('decision_threshold', 0.0)):.2f}" if s1 else "unavailable",
                },
                {
                    "Metric": "Stage 2 validation-selected abstention threshold",
                    "Value": f"{float(s2.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)):.2f}",
                },
            ]
            excluded_perf_keys = {"macro_f1", "confidence_threshold"}
            explicit_performance.extend(
                {
                    "Metric": key.replace("_", " ").title(),
                    "Value": f"{value:.4f}",
                }
                for key, value in perf.items()
                if key not in excluded_perf_keys
            )
            perf_df = pd.DataFrame(explicit_performance)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            st.caption(STAGE2_CONDITIONAL_NOTE)
            st.markdown('</div>', unsafe_allow_html=True)

            # Per-class F1
            pcf1 = mc.get("per_class_f1", {})
            if pcf1:
                st.markdown('<div class="mc-section"><div class="mc-label">Per-Class F1</div>', unsafe_allow_html=True)
                f1_df = pd.DataFrame([
                    {
                        "Class": c,
                        "F1": f"{v['f1']:.4f}",
                        "Support": v.get("support", "—"),
                        "Support status": (
                            low_support_badge(v["support"])
                            if isinstance(v.get("support"), (int, float))
                            else ""
                        ),
                    }
                    for c, v in pcf1.items()
                ])
                styled_f1 = f1_df.style.map(_color_support, subset=["Support status"])
                st.dataframe(
                    styled_f1,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Support status": st.column_config.TextColumn(
                            "Support status",
                            width="medium",
                            help="Metrics with fewer than 50 held-out test samples.",
                        )
                    },
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Limitations
            lims = mc.get("limitations", [])
            if lims:
                st.markdown('<div class="mc-section"><div class="mc-label">Limitations</div>', unsafe_allow_html=True)
                for i, lim in enumerate(lims, 1):
                    st.markdown(f"{i}. {lim}")
                st.markdown('</div>', unsafe_allow_html=True)
