"""
model_card.py — Generate the SemVision model card JSON.

All 13 required sections are populated from training artifacts.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import RANDOM_SEED, DEFAULT_CONFIDENCE_THRESHOLD, STAGE2_CLASSES


def generate_model_card(
    stage1_metrics: dict,
    stage2_metrics: dict,
    split_method_used: str,
    augmentation_winner: str,
    end_to_end_metrics: dict | None = None,
    split_fallback_reason: str | None = None,
) -> dict:
    """
    Generate model card JSON.

    Parameters
    ----------
    stage1_metrics     : full dict from train_stage1 (includes champion_metrics)
    stage2_metrics     : dict from evaluate_cnn on test set
    split_method_used  : 'lot-grouped' | 'stratified'
    augmentation_winner: 'with_augmentation' | 'no_augmentation'

    Returns
    -------
    model_card : dict (also written to artifacts/reports/model_card.json)
    """
    per_class_f1 = {
        cls: {
            "f1":      stage2_metrics["per_class_metrics"][cls]["f1"],
            "support": stage2_metrics["per_class_metrics"][cls]["support"],
        }
        for cls in STAGE2_CLASSES
        if cls in stage2_metrics["per_class_metrics"]
    }

    confidence_threshold = stage2_metrics.get(
        "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
    )
    limitations = [
        "Rare defect classes (e.g., Donut, Near-Full) have significantly fewer "
        "training samples and lower F1.",
        f"Model should abstain (return UNCERTAIN) if max softmax probability "
        f"< {confidence_threshold}.",
        "Grad-CAM provides visual decision support by highlighting wafer regions "
        "influential to the model prediction; process root-cause confirmation "
        "requires manufacturing metadata and engineering investigation.",
    ]

    if split_method_used == "stratified":
        limitations.append(
            "Lot-level grouping was not applied. The exact fallback reason was: "
            f"{split_fallback_reason or 'not recorded'}. This is a sample-level "
            "stratified split, not a leakage-safe lot split; reported test metrics "
            "may be optimistic when wafers from one lot share process patterns."
        )

    ablation = stage2_metrics.get("augmentation_ablation", {})
    if ablation and abs(float(ablation.get("difference", 0.0))) < 0.01:
        limitations.append(
            "Augmentation changed validation macro F1 by less than 0.01; the "
            "simpler no-augmentation model was selected."
        )

    model_card = {
        # ── Required sections (per spec) ─────────────────────────────────────
        "model_name":  "SemVision Stage2 Wafer Pattern CNN",
        "model_type":  (
            "Multi-class CNN for semiconductor wafer map failure pattern classification"
        ),
        "domain_note": (
            "This model classifies spatial defect patterns from categorical wafer maps. "
            "It does NOT perform raw e-beam SEM image defect detection. Wafer maps "
            "represent aggregated die pass/fail outcomes, not inspection imagery."
        ),
        "intended_use": (
            "Yield triage and defect pattern identification from wafer-level test "
            "data in semiconductor manufacturing."
        ),
        "out_of_scope": (
            "Direct use on raw SEM or optical inspection images, root-cause "
            "diagnosis without manufacturing metadata."
        ),
        "training_data": (
            "WM-811K dataset (811,457 wafer maps from real fab production, "
            "Kaggle, CC BY 4.0)"
        ),
        "data_split_method":  split_method_used,
        "data_split_fallback_reason": split_fallback_reason,
        "augmentation_used":  augmentation_winner,
        "performance":            stage2_metrics,
        "per_class_f1":          per_class_f1,
        "limitations":           limitations,
        "random_seed":           RANDOM_SEED,
        "inference_latency_ms":  stage2_metrics["inference_latency_ms"],

        # ── Extra useful fields ───────────────────────────────────────────────
        "worst_class":          stage2_metrics["worst_class"],
        "worst_confusion_pair": stage2_metrics["worst_confusion_pair"],
        "confidence_separation": stage2_metrics.get("confidence_separation", {}),
        "calibration": stage2_metrics.get("calibration", {}),
        "confidence_threshold": confidence_threshold,
        "confidence_threshold_selection": stage2_metrics.get(
            "abstention_validation", {}
        ),
        "stage1_summary": {
            "champion":  stage1_metrics.get("champion", "unknown"),
            "decision_threshold": stage1_metrics.get("decision_threshold"),
            "selection_partition": stage1_metrics.get(
                "champion_selection_partition"
            ),
            "selection_methodology": stage1_metrics.get(
                "champion_selection_methodology"
            ),
            "validation_metrics": stage1_metrics.get(
                "champion_validation_metrics", {}
            ),
            "test_metrics": stage1_metrics.get("champion_metrics", {}),
        },
        "end_to_end_pipeline": end_to_end_metrics or {},
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    reports_dir = os.path.join(project_root, "artifacts", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    card_path = os.path.join(reports_dir, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)

    print(f"Model card saved → {card_path}")
    return model_card
