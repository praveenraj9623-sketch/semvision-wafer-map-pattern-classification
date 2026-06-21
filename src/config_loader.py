"""
config_loader.py — Re-exports all constants from root config.py.

This module bridges the src/ package with the root config.py.
Import from this module in all src/ files.
"""
import sys
import os

# Ensure the project root is on sys.path so `import config` works
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
    RANDOM_SEED,
    WAFER_SIZE,
    STAGE2_CLASSES,
    NUM_DEFECT_CLASSES,
    NONE_CLASS,
    MAX_NONE_SAMPLES,
    MIN_CLASS_SAMPLES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    STAGE1_MAX_FALSE_POSITIVE_RATE,
    MIN_ABSTENTION_COVERAGE,
    CALIBRATION_BINS,
    CNN_EPOCHS,
    CNN_PATIENCE,
    CNN_BATCH_SIZE,
    TEST_SPLIT,
    VAL_SPLIT,
    USE_LOT_GROUPING,
)
