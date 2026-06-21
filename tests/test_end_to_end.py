import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def test_end_to_end_report_contract():
    path = PROJECT_ROOT / "artifacts" / "reports" / "end_to_end_metrics.json"
    if not path.exists():
        pytest.skip("end_to_end_metrics.json not found — run integration training first")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert metrics["evaluation_partition"] == "global_test"
    assert len(metrics["nine_class_system_confusion_matrix"]) == 9
    assert all(len(row) == 9 for row in metrics["nine_class_system_confusion_matrix"])
    assert 0.0 <= metrics["abstention_rate"] <= 1.0
    assert 0.0 <= metrics["end_to_end_defect_recall"] <= 1.0


def test_dataset_summary_records_global_partition():
    path = PROJECT_ROOT / "artifacts" / "reports" / "dataset_summary.json"
    if not path.exists():
        pytest.skip("dataset_summary.json not found — run integration training first")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert sum(summary["global_split_sizes"].values()) == summary["total_samples"]
    assert set(summary["global_class_counts"]["test"]) == set(summary["class_names"])
