# SemVision: Semiconductor Wafer Map Pattern Classification and Triage

> A two-stage ML system for defect pattern triage from semiconductor wafer maps.

## Domain Disclaimer

> This project classifies spatial defect pattern types from semiconductor wafer maps. Wafer maps represent aggregated die pass/fail outcomes across a wafer, not raw e-beam or optical SEM inspection imagery. This is relevant to the wafer inspection and yield analytics ecosystem but is distinct from raw defect image classification.

---

## Business Context

Semiconductor manufacturing generates massive wafer test data. When dies fail, the **spatial pattern of failures** often indicates the underlying process issue:
- **Center patterns** → etch non-uniformity
- **Edge-Ring patterns** → edge-bead removal or spin-coat issues  
- **Scratch patterns** → mechanical handling damage
- **Random patterns** → contamination or particle events

SemVision automates this pattern classification, enabling yield engineers to:
- **Rapidly triage** wafers without manual inspection
- **Route wafers** to the correct engineering team
- **Track pattern trends** over time (process drift detection)
- **Accelerate yield learning** by reducing time-to-root-cause

---

## Architecture

```
WM-811K Wafer Maps (811,457 real fab wafers)
    ↓
Load & Validate (load_wafer_data.py)
  - Drop unlabeled rows (empty failureType)
  - Cap normal wafers at 5,000 for class balance
    ↓
3-Channel Categorical Preprocessing (preprocess.py)
  - Pad to square → nearest-neighbor resize to 64×64
  - Expand to 3 binary channels: [pass, fail, missing]
    ↓
One Global 9-Class Split (split.py)
  - Lot-grouped when all classes retain coverage
  - Exact-reason stratified fallback otherwise
  - Stage 1 uses the global indices unchanged
  - Stage 2 filters defect rows inside those same indices
    ↓
┌──────────────────────────────────────┐
│ Stage 1: Binary Gate                │
│ No Labelled Failure Pattern vs      │
│ Labelled Defect Pattern (RF/LR)     │
│ Validation-selected model + gate    │
└──────────────────────────────────────┘
    ↓ (defect rows only)
┌──────────────────────────────────────┐
│ Stage 2: Pattern CNN (8 classes)    │
│ 3-conv-block + GlobalAvgPool        │
│ Augmentation ablation (±rotation)   │
│ class_weight balanced training      │
└──────────────────────────────────────┘
    ↓
Grad-CAM + Per-Class Evaluation + Model Card
    ↓
Streamlit Demo (precomputed test samples only)
```

---

## Dataset

**WM-811K Dataset**
- **Source corpus:** WM-811K, **811,457 wafer maps**
- Source: [Kaggle — WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- Kaggle mirror license: **CC0: Public Domain**, as listed in the [dataset metadata](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/metadata) (checked 2026-11-01).
- 8 defect pattern classes: `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-Full`, `Random`, `Scratch`
- **Eligible labelled defect-pattern cohort:** 25,519 maps
- **Sampled no-labelled-pattern cohort:** 5,000 maps
- **Final global train / validation / test:** 21,352 / 4,585 / 4,582
- **Unlabelled records are excluded from supervised training and evaluation.**

The modelling cohort is label-enriched for tractable training and rare-pattern evaluation. It includes all eligible labelled defect-pattern wafer maps and a reproducible capped sample of no-labelled-pattern wafers. It is not a direct estimate of natural fab prevalence, production alert volume, precision, or abstention workload.

---

## Key Design Decisions

### 1. 3-Channel Categorical Representation
Wafer maps contain **categorical die states** (0=missing, 1=pass, 2=fail). Rather than treating them as grayscale images (which would be semantically wrong), we convert to 3 binary channels. This preserves the discrete meaning of each die state and enables the CNN to learn from each channel independently.

### 2. Categorical Input and Continuous Explanation Resizing
Since wafer input values are categorical (not continuous intensity), **bilinear or bicubic input interpolation would create invalid fractional values**. Wafer preprocessing therefore uses `skimage.transform.resize(order=0)`. Grad-CAM is a continuous explanation intensity map, so its display overlay is resized bilinearly (`order=1`).

### 3. Leakage-Safe Data Splitting
The pipeline first creates one nine-class global partition. With the audited WM-811K file, `lotName` provides 9,505 valid scalar groups, and the selected split has no lot overlap across train, validation, and test. Stage 1 and Stage 2 are derived from this one partition. If a grouped candidate cannot retain every class, the code falls back to sample-level stratification and records the exact reason; that fallback is explicitly not described as leakage-safe.

### 4. Augmentation Ablation Study
We train two identical CNN variants — with and without discrete 90-degree rotations and horizontal/vertical flips. No continuous interpolation, brightness, blur, zoom, or contrast transforms are used. The winner is selected by validation macro-F1. If the difference is < 0.01, the simpler model is preferred. Results documented in `artifacts/reports/augmentation_ablation.json`.

### 5. Confidence Abstention
The model returns **UNCERTAIN** when max softmax is below a threshold selected exclusively on validation data. The audited run evaluated 0.30–0.90 and selected **0.70** subject to the documented coverage rule. Candidate coverage, macro F1, uncertainty, and error rates are stored in `stage2_metrics.json`.

### 6. MLflow Experiment Tracking
All training runs are logged to MLflow experiments `semvision_stage1` and `semvision_stage2`. Run `mlflow ui` to explore run history, parameters, and metrics.

---

## Performance Metrics

| Metric | Value |
|---|---|
| Stage 1 Champion | Random Forest |
| Stage 1 Validation-Selected Threshold | 0.45 |
| Stage 1 ROC-AUC / PR-AUC | 0.9424 / 0.9889 |
| Stage 1 Defect Gate Recall | 0.9020 |
| Stage 2 Conditional Macro F1 | 0.8412 |
| Stage 2 Validation-Selected Abstention Threshold | 0.70 |
| End-to-End Nine-Class Macro F1 | 0.7484 |
| Stage 2 Worst-Class F1 | 0.6983 (Loc) |
| End-to-End Defect Recall | 0.9020 |
| Abstention Rate / Threshold | 0.1445 / 0.70 |
| Augmentation Winner | With augmentation (+0.0183 validation macro F1) |
| Split Method | Lot-grouped global split |

Stage 2 metrics are conditional on samples routed to Stage 2 by the Stage 1 gate.

---

## Interactive Workflow Diagram

Open the self-contained [SemVision interactive workflow](docs/semvision_workflow.html) by double-clicking the file locally, or publish the `docs/` folder with GitHub Pages.

## Run with Docker

```powershell
docker compose up --build
```

Then open [http://localhost:8501](http://localhost:8501). See the complete [Docker deployment guide](docs/docker_deployment.md) for detached mode, logs, health verification, rebuilds, and deployment scope.

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Dataset
```
data/raw/LSWMD.pkl    ← download from Kaggle and place here
```

### 3. Run Full Training
```bash
python run_training.py
```
*(~30–90 minutes depending on hardware)*

### 4. Run Tests
```bash
# Deterministic tests (no trained artifacts needed):
pytest -m "not integration" -v

# Trained-artifact integration tests:
pytest -m "integration and not ui" -v

# Streamlit page smoke test (run separately for clean teardown):
pytest -m "integration and ui" -v
```

### 5. Launch Streamlit App
```bash
streamlit run app.py
```

### 6. View MLflow Experiments
```bash
mlflow ui --backend-store-uri artifacts/mlruns
```
Navigate to `http://localhost:5000`

---

## Model Card Limitations

- Rare defect classes (Donut, Near-Full) have significantly fewer training samples and lower F1.
- A sample-level stratified fallback is available when viable lot grouping is impossible; such fallback metrics may be optimistic and are not labelled leakage-safe.
- The audited model abstains (returns UNCERTAIN) if max softmax probability is below the validation-selected threshold of 0.70.
- Grad-CAM provides visual decision support by highlighting wafer regions influential to the model prediction; process root-cause confirmation requires manufacturing metadata and engineering investigation.

---

## File Structure

```
SemVision/
├── app.py                          ← Streamlit application (4 pages)
├── run_training.py                 ← Master training runner
├── config.py                       ← All constants (WAFER_SIZE, STAGE2_CLASSES, etc.)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/LSWMD.pkl               ← (gitignored) user-provided dataset
│   ├── processed/                  ← (gitignored) X.npy, labels.npz, splits.npz
│   └── demo/demo_samples.pkl       ← precomputed test samples for Streamlit
│
├── artifacts/
│   ├── models/
│   │   ├── stage1_binary.pkl       ← champion Stage 1 model
│   │   └── stage2_pattern_cnn.h5  ← Stage 2 CNN
│   ├── reports/
│   │   ├── stage1_metrics.json
│   │   ├── stage2_metrics.json
│   │   ├── augmentation_ablation.json
│   │   ├── training_history.json
│   │   └── model_card.json
│   └── mlruns/                     ← (gitignored) MLflow experiment runs
│
├── src/
│   ├── config_loader.py
│   ├── data/
│   │   ├── load_wafer_data.py      ← LSWMD.pkl loader and cleaner
│   │   ├── preprocess.py           ← 3-channel nearest-neighbor resize
│   │   ├── split.py                ← Group/stratified splitting
│   │   └── prepare_demo.py         ← Demo sample precomputation
│   ├── models/
│   │   ├── stage1_model.py         ← LR + RF binary gate
│   │   ├── stage2_cnn.py           ← CNN with augmentation ablation
│   │   ├── evaluate.py             ← Evaluation utilities
│   │   └── model_card.py           ← Model card generator
│   ├── visualization/
│   │   ├── wafer_viz.py            ← Matplotlib + Plotly visualizations
│   │   └── gradcam.py              ← Grad-CAM implementation
│   └── ui/
│       ├── page_overview.py
│       ├── page_classify.py
│       ├── page_analysis.py
│       └── page_evaluation.py
│
└── tests/
    ├── test_preprocess.py          ← No dataset needed
    ├── test_split.py               ← No dataset needed
    ├── test_stage1.py              ← Requires training artifacts
    ├── test_stage2.py              ← Requires training artifacts
    └── test_gradcam.py             ← Requires demo_samples.pkl
```

---

## License

The cited Kaggle mirror lists WM-811K as **CC0: Public Domain**. See the [dataset metadata](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/metadata) and the research paper linked from its data card for source provenance. This statement describes the mirror's listing; it does not assign a license to this repository's code.
