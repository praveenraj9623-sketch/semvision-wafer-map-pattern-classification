"""
app.py — SemVision Streamlit Application.

4 pages:
  1. Overview        — KPIs, class distribution, model comparison
  2. Classify a Wafer — Interactive classification with Grad-CAM
  3. Pattern Analysis — Wafer grid, F1 bars, confusion matrix
  4. Model Evaluation — Full report, training history, model card

Data source: data/demo/demo_samples.pkl + artifacts/reports/*.json
NEVER loads LSWMD.pkl or data/processed/ directly.
"""
import os
import sys

# ── Ensure src/ is on the path ────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SemVision — Wafer Defect Pattern Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": None,
        "About": "SemVision: Semiconductor Wafer Map Pattern Classification & Triage",
    },
)

# ── Global dark-theme CSS ─────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/*
 * Force a deterministic dark canvas. Streamlit otherwise follows a browser-
 * local theme preference, which can make the custom dark cards render on a
 * white page for first-time visitors.
 */
:root {
    color-scheme: dark;
}
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main {
    background: #0e1117 !important;
    color: #f5f7fa !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: rgba(14, 17, 23, 0) !important;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span {
    color: #f5f7fa;
}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
[data-testid="stAppViewContainer"] small {
    color: #94a3b8 !important;
}
[data-testid="stAppViewContainer"] hr {
    border-color: rgba(148, 163, 184, 0.18) !important;
}

/* Inputs and dropdown menus must not inherit a visitor's light theme. */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"],
[data-baseweb="textarea"] {
    background: #171c28 !important;
    border-color: rgba(79, 195, 247, 0.24) !important;
    color: #f5f7fa !important;
}
[data-baseweb="select"] svg,
[data-baseweb="input"] svg {
    fill: #cbd5e1 !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] ul,
[role="listbox"] {
    background: #171c28 !important;
    color: #f5f7fa !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: rgba(79, 195, 247, 0.16) !important;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0e1117 60%, #111520 100%);
    border-right: 1px solid rgba(79,195,247,0.15);
}
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(79,195,247,0.14);
    border-color: rgba(79,195,247,0.55);
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 650;
}

/* Sidebar title */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0 1.2rem 0;
    border-bottom: 1px solid rgba(79,195,247,0.2);
    margin-bottom: 1rem;
}
.sidebar-title {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4fc3f7, #81d4fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.sidebar-subtitle {
    font-size: 0.72rem;
    color: #9e9e9e;
    letter-spacing: 0.05em;
}

/* Main content area */
.main .block-container {
    padding-top: 1.5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1400px;
}

/* Hide default Streamlit header */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

/* Radio nav buttons */
div[data-testid="stSidebar"] .stRadio > div {
    gap: 0.3rem;
}
div[data-testid="stSidebar"] .stRadio label {
    border: 1px solid rgba(79,195,247,0.2);
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s;
    color: #cbd5e1 !important;
    cursor: pointer;
}
div[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(79,195,247,0.12);
    border-color: rgba(79,195,247,0.5);
    color: #fff !important;
}

div[data-testid="stSidebar"] .stRadio label:hover p {
    color: #fff !important;
}

/* Streamlit metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(20,25,40,0.9), rgba(15,18,30,0.95));
    border: 1px solid rgba(79,195,247,0.18);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label { color: #9e9e9e !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #4fc3f7 !important; font-weight: 700; }

/* Dataframes */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1565c0, #1976d2);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
  <div>
    <div class="sidebar-title">🔬 SemVision</div>
    <div class="sidebar-subtitle">Wafer Defect Pattern Classifier</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Classify a Wafer", "🔬 Pattern Analysis", "📋 Model Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### 🏗️ Two-Stage Pipeline")
    st.markdown(
        """
**Stage 1 — Binary Gate**
Logistic Regression vs Random Forest
↓ No Labelled Failure Pattern vs Labelled Defect Pattern

**Stage 2 — Pattern CNN**
8-class convolutional network
↓ Which defect pattern?

**Grad-CAM**
Visual decision regions
(support, not diagnosis)
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚠️ Domain Disclaimer")
    st.caption(
        "This project classifies spatial defect pattern types from semiconductor wafer maps. "
        "Wafer maps represent aggregated die pass/fail outcomes — not raw SEM/optical inspection imagery."
    )
    st.markdown("---")
    st.caption("Dataset: WM-811K · CC BY 4.0")

# ── Page routing ──────────────────────────────────────────────────────────────
from ui.page_overview   import show_overview
from ui.page_classify   import show_classify
from ui.page_analysis   import show_analysis
from ui.page_evaluation import show_evaluation

if "Overview" in page:
    show_overview()
elif "Classify" in page:
    show_classify()
elif "Analysis" in page:
    show_analysis()
elif "Evaluation" in page:
    show_evaluation()
