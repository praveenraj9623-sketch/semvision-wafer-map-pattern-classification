"""
run_training.py — SemVision master training runner.

Usage:
    python run_training.py

Steps:
    1. Load wafer data
    2. Preprocess (3-channel nearest-neighbor resize)
    3. Split (lot-grouped or stratified)
    4. Train Stage 1 (LR vs RF, select champion by ROC-AUC)
    5. Train Stage 2 CNN (augmentation ablation, select winner)
    6. Evaluate both stages
    7. Generate model card
    8. Prepare demo samples for Streamlit app
    9. Print final summary table
"""
import os
import sys
import time
import gc
import logging

# ── Ensure src/ is importable ─────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from data.load_wafer_data  import load_wafer_data
from data.preprocess        import preprocess_dataset
from data.split             import make_splits
from models.stage1_model    import train_stage1
from models.stage2_cnn      import train_stage2_cnn
from models.model_card       import generate_model_card
from models.end_to_end       import evaluate_end_to_end
from data.prepare_demo       import prepare_demo_samples


def _banner(msg: str):
    print("\n" + "═" * 62)
    print(f"  {msg}")
    print("═" * 62)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _banner("SemVision Training Pipeline")
    total_start = time.time()

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[1/9] Loading wafer data …")
    t0 = time.time()
    df_combined = load_wafer_data()
    class_distribution = {
        str(label): int(count)
        for label, count in df_combined["label"].value_counts().items()
    }
    print(f"      Done in {time.time()-t0:.1f}s — {len(df_combined):,} samples loaded")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("\n[2/9] Preprocessing wafer maps …")
    t0 = time.time()
    X, y_binary, y_pattern, lot_groups = preprocess_dataset(df_combined)
    del df_combined
    gc.collect()
    print(f"      Done in {time.time()-t0:.1f}s — X shape: {X.shape}")

    # ── Step 3: Split ─────────────────────────────────────────────────────────
    print("\n[3/9] Creating one global train/val/test split …")
    t0 = time.time()
    splits_dict = make_splits(X, y_binary, y_pattern, lot_groups)
    split_method = splits_dict.get("split_method", "stratified")
    print(f"      Done in {time.time()-t0:.1f}s — method: {split_method}")

    # ── Step 4: Train Stage 1 ─────────────────────────────────────────────────
    print("\n[4/9] Training Stage 1 binary classifier …")
    t0 = time.time()
    stage1_model, stage1_metrics = train_stage1(
        X, y_binary, splits_dict, class_distribution=class_distribution
    )
    print(f"      Done in {time.time()-t0:.1f}s — champion: {stage1_metrics['champion']}")

    # ── Step 5: Train Stage 2 ─────────────────────────────────────────────────
    print("\n[5/9] Training Stage 2 CNN (augmentation ablation) …")
    t0 = time.time()
    stage2_model, stage2_metrics, augmentation_winner = train_stage2_cnn(
        X, y_pattern, splits_dict
    )
    print(f"      Done in {time.time()-t0:.1f}s — winner: {augmentation_winner}")
    # ── Step 6: End-to-end evaluation on the one global test partition ───────
    print("\n[6/9] Evaluating the complete two-stage pipeline …")
    end_to_end_metrics = evaluate_end_to_end(
        X,
        y_binary,
        y_pattern,
        splits_dict,
        stage1_model,
        stage2_model,
        stage1_metrics["decision_threshold"],
        stage2_metrics["confidence_threshold"],
    )
    del stage1_model, stage2_model
    gc.collect()

    # ── Step 7: Model card ────────────────────────────────────────────────────
    print("\n[7/9] Generating model card …")
    t0 = time.time()
    _ = generate_model_card(
        stage1_metrics,
        stage2_metrics,
        split_method,
        augmentation_winner,
        end_to_end_metrics=end_to_end_metrics,
        split_fallback_reason=splits_dict.get("split_fallback_reason") or None,
    )
    print(f"      Done in {time.time()-t0:.1f}s")

    del X, y_binary, y_pattern, lot_groups, splits_dict
    gc.collect()

    # ── Step 8: Prepare demo ──────────────────────────────────────────────────
    print("\n[8/9] Preparing demo samples for Streamlit app …")
    t0 = time.time()
    demo = prepare_demo_samples()
    print(f"      Done in {time.time()-t0:.1f}s — {len(demo)} samples")

    print("\n[9/9] All artifacts generated")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_runtime = time.time() - total_start
    worst_f1 = stage2_metrics["per_class_metrics"][stage2_metrics["worst_class"]]["f1"]

    _banner("TRAINING COMPLETE — FINAL SUMMARY")
    print(f"  {'Stage 1 Champion':<30} {stage1_metrics['champion'].replace('_',' ').title()}")
    print(f"  {'Stage 1 ROC-AUC':<30} {stage1_metrics['champion_metrics']['roc_auc']:.4f}")
    print(f"  {'Stage 2 Macro F1':<30} {stage2_metrics['macro_f1']:.4f}")
    print(f"  {'Stage 2 Weighted F1':<30} {stage2_metrics['weighted_f1']:.4f}")
    print(f"  {'Stage 2 Worst-Class F1':<30} {worst_f1:.4f}  ({stage2_metrics['worst_class']})")
    print(f"  {'Augmentation Winner':<30} {augmentation_winner}")
    print(f"  {'Split Method':<30} {split_method}")
    print(f"  {'Inference Latency':<30} {stage2_metrics['inference_latency_ms']:.1f} ms")
    print(f"  {'End-to-End Macro F1':<30} {end_to_end_metrics['end_to_end_macro_f1']:.4f}")
    print(f"  {'Abstention Threshold':<30} {stage2_metrics['confidence_threshold']:.2f}")
    print(f"  {'Total Runtime':<30} {total_runtime:.0f}s  ({total_runtime/60:.1f} min)")
    print("═" * 62)
    print()
    print("✅ All done! Next steps:")
    print("   streamlit run app.py")
    print("   mlflow ui")
    print("   pytest tests/ -v")
    print()


if __name__ == "__main__":
    main()
