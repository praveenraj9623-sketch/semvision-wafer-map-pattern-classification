RANDOM_SEED = 42
WAFER_SIZE = 64
STAGE2_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-Full", "Random", "Scratch"
]
NUM_DEFECT_CLASSES = 8
NONE_CLASS = "none"
MAX_NONE_SAMPLES = 5000
MIN_CLASS_SAMPLES = 50
# UI/model-card fallback only. The deployed threshold is selected on validation
# data and stored in stage2_metrics.json.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
STAGE1_MAX_FALSE_POSITIVE_RATE = 0.20
MIN_ABSTENTION_COVERAGE = 0.80
CALIBRATION_BINS = 10
CNN_EPOCHS = 20
CNN_PATIENCE = 5
CNN_BATCH_SIZE = 32
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
USE_LOT_GROUPING = True
