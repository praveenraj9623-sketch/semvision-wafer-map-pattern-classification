"""Smoke-test every Streamlit page with the generated production artifacts."""
from pathlib import Path

from streamlit.testing.v1 import AppTest
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.integration, pytest.mark.ui]


def test_all_streamlit_pages_render_without_exceptions():
    required = [
        PROJECT_ROOT / "artifacts" / "reports" / "stage2_metrics.json",
        PROJECT_ROOT / "data" / "demo" / "demo_samples.pkl",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Streamlit production artifacts have not been generated")
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    app.run(timeout=60)
    assert not app.exception

    navigation = app.radio[0]
    assert len(navigation.options) == 4
    expected_copy = {
        "Overview": [
            "Eligible Labelled Defect Maps — Pre-Split Distribution",
            "End-to-End 9-Class Macro F1",
            "Global Train Samples",
            "Stage 2 Defect Validation",
        ],
        "Classify": [
            "Held-out Test Label",
            "This label is available because the demo uses precomputed held-out test samples.",
            "Probability bars are sorted by model confidence",
            "DEFECT PATTERN",
        ],
        "Analysis": [
            "Most Frequent Directional Error",
            "Near-Full has low test support (n=19)",
        ],
        "Evaluation": [
            "Stage 2 Conditional Macro F1",
            "should not be summed again with those outcomes",
            "The modelling cohort is label-enriched for tractable training",
            "they do not change the calibration calculations",
            "Near-Full has low test support (n=19)",
        ],
    }
    for page in navigation.options:
        app.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, f"Streamlit page failed to render: {page}"
        rendered_text = []
        for element_type in ("markdown", "caption", "metric", "info", "warning"):
            for element in getattr(app, element_type):
                rendered_text.extend(
                    [str(getattr(element, "value", "")), str(getattr(element, "label", ""))]
                )
        page_copy = "\n".join(rendered_text)
        page_name = next(name for name in expected_copy if name in page)
        for expected in expected_copy[page_name]:
            assert expected in page_copy, f"Missing expected copy on {page_name}: {expected}"
        if page_name in {"Analysis", "Evaluation"}:
            near_full_warnings = [
                warning
                for warning in app.warning
                if "Near-Full has low test support (n=19)" in str(warning.value)
            ]
            assert len(near_full_warnings) == 1, (
                f"Expected one Near-Full page warning on {page_name}, "
                f"found {len(near_full_warnings)}"
            )

    # Exercise the new Classify filters, including the conditional per-class selector.
    classify_page = next(option for option in app.radio[0].options if "Classify" in option)
    app.radio[0].set_value(classify_page).run(timeout=60)
    filter_widget = next(
        widget for widget in app.selectbox if widget.label == "Filter demo samples"
    )
    assert list(filter_widget.options) == [
        "All samples",
        "Correct predictions",
        "Incorrect predictions",
        "Uncertain predictions",
        "Per-class samples",
    ]
    for filter_mode in filter_widget.options[1:4]:
        next(
            widget for widget in app.selectbox if widget.label == "Filter demo samples"
        ).set_value(filter_mode).run(timeout=60)
        assert not app.exception, f"Classify filter failed: {filter_mode}"
    next(
        widget for widget in app.selectbox if widget.label == "Filter demo samples"
    ).set_value("Per-class samples").run(timeout=60)
    assert any(widget.label == "Held-out test class" for widget in app.selectbox)
    assert not app.exception

    # Exercise both confusion-matrix rendering paths.
    analysis_page = next(option for option in app.radio[0].options if "Analysis" in option)
    app.radio[0].set_value(analysis_page).run(timeout=60)
    next(
        widget for widget in app.selectbox if widget.label == "Confusion matrix view"
    ).set_value("Raw counts").run(timeout=60)
    assert not app.exception
