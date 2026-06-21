"""Deterministic tests for portfolio-release UI reporting helpers."""

import numpy as np

from ui.page_classify import (
    FILTER_OPTIONS,
    _confidence_bar_plotly,
    _filter_demo_samples,
)
from ui.reporting_copy import (
    COHORT_TRANSPARENCY_STATEMENT,
    cohort_summary_rows,
    low_support_badge,
    low_support_classes,
    low_support_warning,
)
from visualization.wafer_viz import plot_confusion_matrix_plotly


def _demo_sample(**overrides):
    sample = {
        "is_defect": 1,
        "stage1_prediction": 1,
        "predicted_class": "Center",
        "true_class": "Center",
        "stage2_confidences": np.array([0.8, 0.2]),
        "stage2_confidence": 0.8,
        "confidence_threshold": 0.7,
    }
    sample.update(overrides)
    return sample


def test_demo_filter_modes_are_supported_and_mutually_decisive():
    samples = [
        _demo_sample(),
        _demo_sample(predicted_class="Loc"),
        _demo_sample(stage2_confidence=0.6),
        _demo_sample(is_defect=0, true_class="none", stage1_prediction=0),
        _demo_sample(true_class="Scratch", stage1_prediction=0),
    ]
    assert FILTER_OPTIONS == [
        "All samples",
        "Correct predictions",
        "Incorrect predictions",
        "Uncertain predictions",
        "Per-class samples",
    ]
    assert len(_filter_demo_samples(samples, "All samples")) == 5
    assert len(_filter_demo_samples(samples, "Correct predictions")) == 2
    assert len(_filter_demo_samples(samples, "Incorrect predictions")) == 2
    assert len(_filter_demo_samples(samples, "Uncertain predictions")) == 1
    assert len(_filter_demo_samples(samples, "Per-class samples", "Scratch")) == 1


def test_probability_bars_are_descending_and_prediction_is_highlighted():
    figure = _confidence_bar_plotly(
        ["Center", "Donut", "Loc"],
        [0.2, 0.7, 0.1],
        predicted_idx=1,
        confidence_threshold=0.6,
    )
    assert list(figure.data[0].x) == [0.7, 0.2, 0.1]
    assert list(figure.data[0].y) == ["Donut", "Center", "Loc"]
    assert list(figure.data[0].marker.color)[0] == "#4fc3f7"
    assert any(float(shape.x0) == 0.6 for shape in figure.layout.shapes)


def test_confusion_matrix_supports_normalized_and_raw_views():
    matrix = np.array([[9, 1], [2, 8]])
    classes = ["Center", "Loc"]
    normalized = plot_confusion_matrix_plotly(matrix, classes, "Row-normalized %")
    raw = plot_confusion_matrix_plotly(matrix, classes, "Raw counts")
    np.testing.assert_allclose(np.asarray(normalized.data[0].z), [[0.9, 0.1], [0.2, 0.8]])
    np.testing.assert_array_equal(np.asarray(raw.data[0].z), matrix)
    assert normalized.layout.annotations[0].text == "90.0%"
    assert raw.layout.annotations[0].text == "9"


def test_cohort_summary_is_derived_from_dataset_counts():
    dataset = {
        "global_class_counts": {
            "all": {"none": 5000, "Center": 10_000, "Scratch": 15_519}
        },
        "global_split_sizes": {"train": 21_352, "validation": 4_585, "test": 4_582},
    }
    rows = cohort_summary_rows(dataset)
    values = {row["Cohort fact"]: row["Value"] for row in rows}
    assert values["Source corpus"] == "WM-811K, 811,457 wafer maps"
    assert values["Eligible labelled defect-pattern cohort"] == "25,519 maps"
    assert values["Sampled no-labelled-pattern cohort"] == "5,000 maps"
    assert values["Final global train / validation / test"] == "21,352 / 4,585 / 4,582"
    assert COHORT_TRANSPARENCY_STATEMENT.endswith("abstention workload.")


def test_low_support_helpers_use_recorded_support_and_strict_threshold():
    metrics = {
        "Near-Full": {"f1": 0.8947, "support": 19},
        "Small-Class": {"f1": 0.7, "support": 49},
        "Boundary-Class": {"f1": 0.8, "support": 50},
    }
    assert low_support_classes(metrics) == [
        {"class_name": "Near-Full", "support": 19},
        {"class_name": "Small-Class", "support": 49},
    ]
    assert low_support_badge(19) == "Low support (n=19)"
    assert low_support_badge(50) == ""
    assert low_support_warning("Near-Full", 19) == (
        "⚠️ Near-Full has low test support (n=19). Interpret this F1 score with "
        "caution; it has higher uncertainty and should not be compared directly with "
        "higher-support classes."
    )
