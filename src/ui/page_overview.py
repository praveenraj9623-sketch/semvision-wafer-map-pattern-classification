"""
page_overview.py — SemVision Overview Page.

Shows:
  - KPI metric cards (glassmorphism style via custom CSS)
  - Defect class distribution chart (Plotly)
  - Stage 1 LR vs RF comparison table
  - Augmentation ablation result card
"""
import os
import sys
import json
import pickle
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Ensure project root on sys.path
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from config_loader import DEFAULT_CONFIDENCE_THRESHOLD, STAGE2_CLASSES
from ui.reporting_copy import COHORT_TRANSPARENCY_STATEMENT, cohort_summary_rows


# ─── CSS injection ─────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin: 1.2rem 0 2rem 0;
}
@media (max-width: 1000px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.kpi-card {
    background: linear-gradient(135deg, rgba(30,35,50,0.9) 0%, rgba(20,25,40,0.95) 100%);
    border: 1px solid rgba(79,195,247,0.25);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(79,195,247,0.15);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9e9e9e;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 2.0rem;
    font-weight: 700;
    color: #4fc3f7;
    line-height: 1.1;
}
.kpi-value.green { color: #66bb6a; }
.kpi-value.red   { color: #ef5350; }
.kpi-value.gold  { color: #ffd54f; }
.kpi-note {
    margin-top: 0.55rem;
    color: #94a3b8;
    font-size: 0.68rem;
    line-height: 1.35;
}

.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    margin: 2rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(79,195,247,0.3);
}
.champion-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(90deg, #1b5e20, #2e7d32);
    border: 1px solid #43a047;
    border-radius: 30px;
    padding: 0.5rem 1.2rem;
    font-weight: 700;
    color: #a5d6a7;
    font-size: 0.95rem;
}
.ablation-card {
    background: linear-gradient(135deg, rgba(25,30,45,0.95), rgba(18,22,35,0.98));
    border: 1px solid rgba(255,213,79,0.3);
    border-radius: 14px;
    padding: 1.4rem;
    margin: 1rem 0;
}
.ablation-winner {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffd54f;
    text-align: center;
    margin-top: 0.8rem;
}
.banner {
    background: linear-gradient(135deg, rgba(13,71,161,0.4), rgba(25,118,210,0.3));
    border-left: 4px solid #4fc3f7;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #90caf9;
    margin-bottom: 1.5rem;
}
.cohort-note {
    background: rgba(79,195,247,0.06);
    border: 1px solid rgba(79,195,247,0.18);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.55;
}
</style>
"""


@st.cache_data
def _load_reports():
    """Load all report JSON files."""
    reports_dir = os.path.join(_project_root, "artifacts", "reports")
    paths = {
        "stage1": os.path.join(reports_dir, "stage1_metrics.json"),
        "stage2": os.path.join(reports_dir, "stage2_metrics.json"),
        "ablation": os.path.join(reports_dir, "augmentation_ablation.json"),
        "dataset": os.path.join(reports_dir, "dataset_summary.json"),
        "e2e": os.path.join(reports_dir, "end_to_end_metrics.json"),
    }
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path) as f:
                data[key] = json.load(f)
        else:
            data[key] = None
    return data


@st.cache_data
def _load_demo():
    demo_path = os.path.join(_project_root, "data", "demo", "demo_samples.pkl")
    if os.path.exists(demo_path):
        with open(demo_path, "rb") as f:
            return pickle.load(f)
    return []


def show_overview():
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="banner">ℹ️ <strong>Data:</strong> WM-811K semiconductor wafer maps (CC BY 4.0). '
        'This demo uses precomputed test-set samples. '
        'SemVision classifies wafer-level die patterns — not raw SEM/optical inspection images.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("# 🔬 SemVision — Wafer Pattern Classification")
    st.markdown(
        "A two-stage ML system for semiconductor wafer defect pattern triage — "
        "from a binary No Labelled Failure Pattern vs Labelled Defect Pattern gate "
        "to 8-class pattern identification."
    )

    reports = _load_reports()
    demo    = _load_demo()

    if not all(reports.values()):
        st.warning(
            "⚠️ Training artifacts not found. Run `python run_training.py` first, "
            "then `python src/data/prepare_demo.py`."
        )
        return

    s1 = reports["stage1"]
    s2 = reports["stage2"]
    ab = reports["ablation"]
    dataset = reports["dataset"]
    e2e = reports["e2e"]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

    roc_auc   = s1["champion_metrics"]["roc_auc"]
    macro_f1  = s2["macro_f1"]
    end_to_end_macro_f1 = e2e["end_to_end_macro_f1"]
    worst_f1  = s2["per_class_metrics"][s2["worst_class"]]["f1"]
    n_defect_cls = len(s2["per_class_metrics"])
    split_sizes = dataset["global_split_sizes"]
    n_train = int(split_sizes["train"])
    confidence_threshold = float(
        s2.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    )

    roc_cls   = "green" if roc_auc  >= 0.90 else ("gold" if roc_auc  >= 0.80 else "red")
    f1_cls    = "green" if macro_f1 >= 0.70 else ("gold" if macro_f1 >= 0.50 else "red")
    wf1_cls   = "green" if worst_f1 >= 0.50 else "red"

    kpi_html = f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Global Train Samples</div>
        <div class="kpi-value">{n_train:,}</div>
        <div class="kpi-note">Includes the sampled no-labelled-pattern cohort and labelled defect-pattern cohort.</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Defect Classes</div>
        <div class="kpi-value">{n_defect_cls}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Stage 1 ROC-AUC</div>
        <div class="kpi-value {roc_cls}">{roc_auc:.4f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Stage 2 Conditional Macro F1</div>
        <div class="kpi-value {f1_cls}">{macro_f1:.4f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">End-to-End 9-Class Macro F1</div>
        <div class="kpi-value {f1_cls}">{end_to_end_macro_f1:.4f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Worst-Class F1</div>
        <div class="kpi-value {wf1_cls}">{worst_f1:.4f} ({s2["worst_class"]})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Confidence Threshold</div>
        <div class="kpi-value gold">{confidence_threshold:.2f}</div>
      </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🧾 Modelling Cohort</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cohort-note">{COHORT_TRANSPARENCY_STATEMENT}</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(cohort_summary_rows(dataset)),
        use_container_width=True,
        hide_index=True,
    )

    # ── Class Distribution ────────────────────────────────────────────────────
    distribution_title = "Eligible Labelled Defect Maps — Pre-Split Distribution"
    st.markdown(
        f'<div class="section-header">📈 {distribution_title}</div>',
        unsafe_allow_html=True,
    )

    class_counts = {
        class_name: int(count)
        for class_name, count in s1.get("class_distribution", {}).items()
        if class_name in STAGE2_CLASSES
    }

    if class_counts:
        from visualization.wafer_viz import plot_class_distribution_plotly
        st.plotly_chart(
            plot_class_distribution_plotly(class_counts, title=distribution_title),
            use_container_width=True,
        )
        st.caption(
            f"All {sum(class_counts.values()):,} eligible labelled defect maps before splitting. "
            "This distribution is not the global training-partition count shown in the KPI above."
        )
    else:
        st.info("Class distribution was not recorded in the Stage 1 report.")

    split_counts = dataset["global_class_counts"]
    stage2_train = sum(
        int(count) for label, count in split_counts["train"].items() if label != "none"
    )
    stage2_validation = sum(
        int(count)
        for label, count in split_counts["validation"].items()
        if label != "none"
    )
    stage2_test = sum(
        int(count) for label, count in split_counts["test"].items() if label != "none"
    )
    split_cols = [*st.columns(3), *st.columns(3)]
    for column, label, value in zip(
        split_cols,
        [
            "Global Train",
            "Global Validation",
            "Global Test",
            "Stage 2 Defect Train",
            "Stage 2 Defect Validation",
            "Stage 2 Defect Test",
        ],
        [
            split_sizes["train"],
            split_sizes["validation"],
            split_sizes["test"],
            stage2_train,
            stage2_validation,
            stage2_test,
        ],
    ):
        column.metric(label, f"{int(value):,}")

    # ── Stage 1 Comparison ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚖️ Stage 1 Model Comparison</div>', unsafe_allow_html=True)

    lr_m  = s1["logistic_regression_metrics"]
    rf_m  = s1["random_forest_metrics"]
    champ = s1["champion"]

    metrics_list = [
        "roc_auc", "pr_auc", "defect_pattern_recall",
        "defect_pattern_precision", "false_negative_rate",
        "false_positive_rate", "f1",
    ]
    labels_map   = {"accuracy":"Accuracy", "precision":"Precision",
                    "recall":"Recall", "f1":"F1", "roc_auc":"ROC-AUC",
                    "pr_auc":"PR-AUC",
                    "defect_pattern_recall":"Defect Recall",
                    "defect_pattern_precision":"Defect Precision",
                    "false_negative_rate":"False-Negative Rate",
                    "false_positive_rate":"False-Positive Rate"}

    df_cmp = pd.DataFrame({
        "Metric":              [labels_map[m] for m in metrics_list],
        "Logistic Regression": [f"{lr_m[m]:.4f}" for m in metrics_list],
        "Random Forest":       [f"{rf_m[m]:.4f}" for m in metrics_list],
    })
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)
    st.caption(
        "Champion and decision threshold were selected on validation data using "
        "the recall-prioritized gate rule; this table reports held-out test metrics."
    )

    champ_display = champ.replace("_", " ").title()
    st.markdown(
        f'<div style="text-align:center;margin:0.5rem 0;">'
        f'<span class="champion-badge">🏆 Champion: {champ_display}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Augmentation Ablation ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧪 Augmentation Ablation Study</div>', unsafe_allow_html=True)

    diff   = ab["difference"]
    winner = ab.get("winner", "no_augmentation")
    note   = ab.get("note", "")

    col_a, col_b, col_diff = st.columns(3)
    col_a.metric("Variant A — No Augmentation",  f"{ab['variant_a_val_macro_f1']:.4f}")
    col_b.metric("Variant B — With Augmentation", f"{ab['variant_b_val_macro_f1']:.4f}")
    col_diff.metric("Δ Macro F1", f"{diff:+.4f}")

    winner_txt = (
        "Variant B (with augmentation)"
        if winner == "with_augmentation"
        else "Variant A (no augmentation)"
    )
    win_color = "#ffd54f"
    ablation_html = f"""
    <div class="ablation-card">
      <div class="ablation-winner">🏆 Winner: {winner_txt}</div>
      {"<p style='text-align:center;color:#9e9e9e;font-size:0.82rem;margin-top:0.5rem;'>" + note + "</p>" if note else ""}
    </div>
    """
    st.markdown(ablation_html, unsafe_allow_html=True)
