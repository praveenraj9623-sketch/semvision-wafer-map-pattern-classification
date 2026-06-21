"""Shared portfolio-release reporting copy derived from existing artifacts."""

SOURCE_CORPUS_COUNT = 811_457
LOW_SUPPORT_THRESHOLD = 50

COHORT_TRANSPARENCY_STATEMENT = (
    "The modelling cohort is label-enriched for tractable training and rare-pattern "
    "evaluation. It includes all eligible labelled defect-pattern wafer maps and a "
    "reproducible capped sample of no-labelled-pattern wafers. It is not a direct "
    "estimate of natural fab prevalence, production alert volume, precision, or "
    "abstention workload."
)

AUGMENTATION_DESCRIPTION = (
    "Discrete 90-degree rotations and horizontal/vertical flips; no continuous "
    "interpolation, brightness, blur, zoom, or contrast transforms."
)

STAGE2_CONDITIONAL_NOTE = (
    "Stage 2 metrics are conditional on samples routed to Stage 2 by the Stage 1 gate."
)


def low_support_classes(
    per_class_metrics: dict,
    threshold: int = LOW_SUPPORT_THRESHOLD,
) -> list[dict]:
    """Return classes whose recorded test support is below the UI caution threshold."""
    flagged = []
    for class_name, metrics in per_class_metrics.items():
        support = metrics.get("support") if isinstance(metrics, dict) else None
        if isinstance(support, (int, float)) and not isinstance(support, bool):
            support = int(support)
            if support < threshold:
                flagged.append({"class_name": class_name, "support": support})
    return flagged


def low_support_badge(support: int, threshold: int = LOW_SUPPORT_THRESHOLD) -> str:
    """Return compact table copy for a low-support metric, otherwise an empty string."""
    support = int(support)
    return f"Low support (n={support})" if support < threshold else ""


def low_support_warning(class_name: str, support: int) -> str:
    """Return the accessible full warning shown once on each relevant page."""
    return (
        f"⚠️ {class_name} has low test support (n={int(support)}). Interpret this F1 "
        "score with caution; it has higher uncertainty and should not be compared "
        "directly with higher-support classes."
    )


def cohort_summary_rows(dataset_summary: dict) -> list[dict]:
    """Return display rows from the immutable dataset summary artifact."""
    all_counts = dataset_summary["global_class_counts"]["all"]
    split_sizes = dataset_summary["global_split_sizes"]
    eligible_defects = sum(
        int(count) for label, count in all_counts.items() if label != "none"
    )
    sampled_none = int(all_counts["none"])
    return [
        {"Cohort fact": "Source corpus", "Value": f"WM-811K, {SOURCE_CORPUS_COUNT:,} wafer maps"},
        {
            "Cohort fact": "Eligible labelled defect-pattern cohort",
            "Value": f"{eligible_defects:,} maps",
        },
        {
            "Cohort fact": "Sampled no-labelled-pattern cohort",
            "Value": f"{sampled_none:,} maps",
        },
        {
            "Cohort fact": "Final global train / validation / test",
            "Value": (
                f"{int(split_sizes['train']):,} / {int(split_sizes['validation']):,} / "
                f"{int(split_sizes['test']):,}"
            ),
        },
        {
            "Cohort fact": "Unlabelled records",
            "Value": "Excluded from supervised training and evaluation",
        },
    ]


def cohort_summary_html(dataset_summary: dict) -> str:
    """Return a compact HTML list suitable for the rendered Model Card."""
    return "<br>".join(
        f"<strong>{row['Cohort fact']}:</strong> {row['Value']}"
        for row in cohort_summary_rows(dataset_summary)
    )
