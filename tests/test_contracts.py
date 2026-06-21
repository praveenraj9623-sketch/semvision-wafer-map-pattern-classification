"""Fast contract checks for data labels, UI paths, and model architecture."""
from pathlib import Path

import numpy as np

from data.load_wafer_data import _extract_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wm811k_label_shapes_are_canonicalized():
    assert _extract_label(np.array([["Center"]])) == "Center"
    assert _extract_label(np.array([["Near-full"]])) == "Near-Full"
    assert _extract_label(["none"]) == "none"
    assert _extract_label(np.empty((0, 0))) is None
    assert _extract_label(np.array([["Center", "Donut"]])) is None


def test_ui_modules_resolve_the_workspace_root():
    from ui import page_analysis, page_classify, page_evaluation, page_overview

    for module in (page_analysis, page_classify, page_evaluation, page_overview):
        assert Path(module._project_root).resolve() == PROJECT_ROOT


def test_streamlit_source_does_not_load_raw_or_processed_artifacts():
    app_sources = [PROJECT_ROOT / "app.py", *(PROJECT_ROOT / "src" / "ui").glob("*.py")]
    source = "\n".join(path.read_text(encoding="utf-8") for path in app_sources)
    assert "pickle.load" in source
    assert "LSWMD.pkl" not in source.replace("NEVER loads LSWMD.pkl", "")
    assert '"processed"' not in source


def test_streamlit_theme_is_pinned_dark():
    config = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'base = "dark"' in config
    assert 'backgroundColor = "#0E1117"' in config
    assert 'textColor = "#F5F7FA"' in config


def test_dashboard_uses_polished_failure_pattern_terminology():
    sources = [
        PROJECT_ROOT / "app.py",
        *(PROJECT_ROOT / "src" / "ui").glob("*.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "Normal vs Defect" not in source
    assert "Normal/Defect" not in source
    assert "True Class" not in (PROJECT_ROOT / "src" / "ui" / "page_classify.py").read_text(
        encoding="utf-8"
    )
    assert "Held-out Test Label" in source
    assert "Eligible Labelled Defect Maps — Pre-Split Distribution" in source


def test_end_to_end_display_outcomes_are_mutually_exclusive():
    from ui.page_evaluation import _build_end_to_end_display

    report = {
        "global_test_samples": 4582,
        "counts": {
            "correctly_identified_no_labelled_pattern": 581,
            "no_labelled_pattern_sent_to_stage2": 154,
            "defect_pattern_missed_by_stage1": 377,
            "correctly_classified_defect_pattern": 2746,
            "defect_pattern_sent_to_stage2_but_misclassified": 149,
            "uncertain_predictions": 662,
            "uncertain_true_defect": 575,
            "uncertain_true_none": 87,
        },
    }
    outcomes, uncertainty = _build_end_to_end_display(report)
    assert sum(row["Count"] for row in outcomes) == 4582
    assert uncertainty[0]["Count"] == uncertainty[1]["Count"] + uncertainty[2]["Count"]
    assert outcomes[1]["Count"] == 154 - 87
    assert outcomes[1]["Final routing outcome"] == (
        "Stage 1 false positive → Stage 2 decisive prediction"
    )
    assert outcomes[2]["Final routing outcome"] == (
        "Stage 1 false negative → defect-pattern missed"
    )


def test_release_copy_is_present_without_editing_report_artifacts():
    from ui.reporting_copy import COHORT_TRANSPARENCY_STATEMENT

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    ui_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "src" / "ui").glob("*.py")
    )
    assert COHORT_TRANSPARENCY_STATEMENT in readme
    assert "Global Train Samples" in ui_source
    assert "Stage 2 Defect Validation" in ui_source
    assert "View raw record payload" in ui_source
    assert "Low support" in ui_source
    assert "Confidence-Bin Sample Counts" in ui_source


def test_stage2_architecture_matches_specification():
    from models.stage2_cnn import build_cnn

    model = build_cnn()
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    assert layer_types == [
        "Conv2D", "BatchNormalization", "MaxPooling2D",
        "Conv2D", "BatchNormalization", "MaxPooling2D",
        "Conv2D", "BatchNormalization", "GlobalAveragePooling2D",
        "Dense", "Dropout", "Dense",
    ]
    assert model.input_shape == (None, 64, 64, 3)
    assert model.output_shape == (None, 8)
