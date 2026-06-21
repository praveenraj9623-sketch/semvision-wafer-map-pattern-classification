"""
load_wafer_data.py — Load and clean WM-811K wafer map data from LSWMD.pkl.

DESIGN NOTES:
- failureType values are lists: ['Center'], ['none'], or []
- Empty list [] and NaN are UNLABELED — excluded completely
- normal wafers (label == 'none') are capped at MAX_NONE_SAMPLES for class balance
- Saves metadata (no waferMap arrays) to data/processed/metadata.csv
"""
import logging
import json
import pickle
import os
import sys
import pandas as pd
import numpy as np

# Ensure src/ on path for config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import (
    STAGE2_CLASSES, NONE_CLASS, MAX_NONE_SAMPLES, MIN_CLASS_SAMPLES, RANDOM_SEED
)


LOGGER = logging.getLogger(__name__)


def _load_pickle_compat(data_path: str) -> pd.DataFrame:
    """Load the legacy Python-2/pandas pickle used by WM-811K.

    The commonly distributed ``LSWMD.pkl`` references the pre-pandas-0.20
    ``pandas.indexes`` package and contains Python-2 strings.  The aliases are
    deliberately local to loading this trusted dataset; no custom unpickler or
    alternate data source is used.
    """
    import pandas.core.indexes as pandas_indexes
    import pandas.core.indexes.base as pandas_indexes_base
    import pandas.core.indexes.range as pandas_indexes_range

    sys.modules.setdefault("pandas.indexes", pandas_indexes)
    sys.modules.setdefault("pandas.indexes.base", pandas_indexes_base)
    sys.modules.setdefault("pandas.indexes.range", pandas_indexes_range)

    with open(data_path, "rb") as file_obj:
        return pickle.load(file_obj, encoding="latin1")


def _extract_label(failure_type):
    """Return one canonical label or ``None`` for an unlabeled/malformed row."""
    if failure_type is None:
        return None
    if isinstance(failure_type, float) and np.isnan(failure_type):
        return None

    if isinstance(failure_type, str):
        raw_label = failure_type.strip()
    elif isinstance(failure_type, (list, tuple, np.ndarray)):
        values = np.asarray(failure_type, dtype=object).reshape(-1)
        if values.size != 1:
            return None
        raw_label = str(values[0]).strip()
    else:
        return None

    canonical = {label.casefold(): label for label in STAGE2_CLASSES}
    canonical[NONE_CLASS.casefold()] = NONE_CLASS
    # The released WM-811K pickle spells this value ``Near-full``.
    canonical["near-full"] = "Near-Full"
    return canonical.get(raw_label.casefold())


def load_wafer_data() -> pd.DataFrame:
    """
    Load wafer data from data/raw/LSWMD.pkl.

    Returns
    -------
    df_combined : DataFrame
        Columns: all original columns + 'label', 'is_defect'.
        Excludes unlabeled rows. Includes both defect and (sampled) none wafers.
    """
    # ── Resolve path relative to this file ──────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "raw", "LSWMD.pkl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"LSWMD.pkl not found at: {data_path}\n"
            "Please place the dataset file at data/raw/LSWMD.pkl"
        )

    print(f"Loading dataset from: {data_path}")
    df = _load_pickle_compat(data_path)

    required_columns = {"waferMap", "failureType"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # ── Verify columns ───────────────────────────────────────────────────────
    print(f"\nDataFrame columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst row sample:")
    print(df.iloc[0])
    print()
    LOGGER.info("Loaded WM-811K shape=%s columns=%s", df.shape, df.columns.tolist())
    LOGGER.info("First WM-811K row: %s", df.iloc[0].to_dict())

    # ── Drop unlabeled rows (empty list [] or NaN) ───────────────────────────
    n_before = len(df)
    df["label"] = df["failureType"].map(_extract_label)
    df = df[df["label"].notna()].copy()
    n_dropped = n_before - len(df)
    print(f"Dropped {n_dropped:,} unlabeled rows ({n_dropped/n_before*100:.1f}%)")

    # ── Defect subset ────────────────────────────────────────────────────────
    df_defect = df[df["label"].isin(STAGE2_CLASSES)].copy()

    # Drop any defect class with fewer than MIN_CLASS_SAMPLES
    class_counts = df_defect["label"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_CLASS_SAMPLES].index.tolist()
    dropped_classes = [c for c in STAGE2_CLASSES if c not in valid_classes]
    if dropped_classes:
        print(f"Dropping rare defect classes (< {MIN_CLASS_SAMPLES} samples): {dropped_classes}")
    df_defect = df_defect[df_defect["label"].isin(valid_classes)].copy()

    # ── Normal subset (capped) ───────────────────────────────────────────────
    df_none = df[df["label"] == NONE_CLASS].copy()
    if len(df_none) > MAX_NONE_SAMPLES:
        df_none = df_none.sample(n=MAX_NONE_SAMPLES, random_state=RANDOM_SEED)
        print(f"Sampled {MAX_NONE_SAMPLES:,} normal wafers (capped from {len(df[df['label'] == NONE_CLASS]):,})")

    # ── Combine ──────────────────────────────────────────────────────────────
    df_combined = pd.concat([df_defect, df_none], axis=0).copy()
    df_combined["is_defect"] = (df_combined["label"] != NONE_CLASS).astype(int)
    df_combined["raw_failure_type"] = df_combined["failureType"].map(
        lambda value: json.dumps(np.asarray(value).tolist())
    )
    original_shapes = df_combined["waferMap"].map(lambda value: np.asarray(value).shape)
    df_combined["original_height"] = original_shapes.map(lambda shape: int(shape[0]))
    df_combined["original_width"] = original_shapes.map(lambda shape: int(shape[1]))

    # ── Print class distribution ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'CLASS DISTRIBUTION':^55}")
    print("=" * 55)
    dist = df_combined["label"].value_counts()
    total = len(df_combined)
    for cls, cnt in dist.items():
        bar = "#" * int(cnt / total * 30)
        print(f"  {cls:<15} {cnt:>6,}  ({cnt/total*100:5.1f}%)  {bar}")
    print(f"\n  {'TOTAL':<15} {total:>6,}")
    print("=" * 55)

    # ── Save metadata CSV (no waferMap arrays) ───────────────────────────────
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    metadata_cols = [
        "label", "is_defect", "raw_failure_type",
        "original_height", "original_width",
    ]
    if "lotName" in df_combined.columns:
        metadata_cols.append("lotName")

    metadata_path = os.path.join(processed_dir, "metadata.csv")
    metadata = df_combined[metadata_cols].copy()
    metadata.index.name = "source_index"
    metadata.to_csv(metadata_path)
    print(f"\nMetadata saved to: {metadata_path}")

    return df_combined


if __name__ == "__main__":
    df = load_wafer_data()
    print(f"\nLoaded {len(df):,} samples successfully.")
