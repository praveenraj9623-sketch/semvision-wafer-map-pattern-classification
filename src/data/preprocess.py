"""
preprocess.py — Categorical wafer map preprocessing.

CRITICAL RULE:
Wafer maps contain categorical die states (0=missing/edge, 1=pass, 2=fail).
They are NOT grayscale photos. Never use bilinear, bicubic, or any
interpolation that creates fractional values.

resize_wafer_map:
  1. Pad to square (constant_values=0, nearest-neighbor)
  2. Resize to WAFER_SIZE x WAFER_SIZE with order=0 (nearest-neighbor)
  3. Round to int8 to eliminate any rounding artifacts
  4. Expand to 3 binary channels: [pass_mask, fail_mask, missing_mask]

preprocess_dataset:
  Applies resize_wafer_map to all rows, assembles X, y_binary, y_pattern,
  lot_groups; saves to data/processed/.
"""
import os
import sys
import numpy as np
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import WAFER_SIZE, STAGE2_CLASSES, NONE_CLASS


def resize_wafer_map(wafer_array) -> np.ndarray:
    """
    Resize a 2-D wafer map to (WAFER_SIZE, WAFER_SIZE, 3).

    Parameters
    ----------
    wafer_array : array-like, shape (H, W)
        Integer array with values 0 (missing), 1 (pass), 2 (fail).

    Returns
    -------
    np.ndarray, shape (64, 64, 3), dtype float32
        Channel 0: pass-die mask  (wafer == 1)
        Channel 1: fail-die mask  (wafer == 2)
        Channel 2: missing-die mask (wafer == 0)
    """
    arr = np.asarray(wafer_array)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError(f"wafer_array must be a non-empty 2-D array; got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise ValueError("wafer_array must contain finite numeric die states")
    invalid_states = np.setdiff1d(np.unique(arr), np.array([0, 1, 2]))
    if invalid_states.size:
        raise ValueError(f"wafer_array contains invalid die states: {invalid_states.tolist()}")
    arr = arr.astype(np.int8, copy=False)

    # ── 1. Pad to square ─────────────────────────────────────────────────────
    h, w = arr.shape
    max_dim = max(h, w)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left

    arr_padded = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,  # 0 = missing die — safe edge fill
    )

    # ── 2. Nearest-neighbor resize ───────────────────────────────────────────
    arr_resized = resize(
        arr_padded,
        (WAFER_SIZE, WAFER_SIZE),
        order=0,              # nearest-neighbor — MANDATORY for categorical data
        anti_aliasing=False,
        preserve_range=True,
    )

    # ── 3. Snap to int8 to eliminate any float artifacts ────────────────────
    arr_resized = np.round(arr_resized).astype(np.int8)

    # ── 4. Three binary channels ─────────────────────────────────────────────
    ch_pass    = (arr_resized == 1).astype(np.float32)
    ch_fail    = (arr_resized == 2).astype(np.float32)
    ch_missing = (arr_resized == 0).astype(np.float32)

    return np.stack([ch_pass, ch_fail, ch_missing], axis=-1)


def preprocess_dataset(df_combined):
    """
    Preprocess the full combined DataFrame.

    Parameters
    ----------
    df_combined : pd.DataFrame
        Must have columns: waferMap, label, is_defect.
        Optionally: lotName.

    Returns
    -------
    X         : np.ndarray (N, 64, 64, 3)  float32
    y_binary  : np.ndarray (N,)            int   0=none, 1=defect
    y_pattern : np.ndarray (N,)            int   0-7=defect class, -1=none
    lot_groups: np.ndarray (N,) or None
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    print(f"Preprocessing {len(df_combined):,} wafer maps …")

    wafer_maps = []
    for i, wm in enumerate(df_combined["waferMap"].values):
        wafer_maps.append(resize_wafer_map(wm))
        if (i + 1) % 1000 == 0:
            print(f"  … processed {i+1:,} / {len(df_combined):,}", end="\r")

    print(f"  … processed {len(df_combined):,} / {len(df_combined):,}")

    X = np.stack(wafer_maps)            # (N, 64, 64, 3)
    print(f"X shape: {X.shape},  dtype: {X.dtype}")

    # ── Binary labels ────────────────────────────────────────────────────────
    y_binary = df_combined["is_defect"].values.astype(np.int32)

    # ── Pattern labels: 0-7 for defects, -1 for none ────────────────────────
    label_to_idx = {cls: idx for idx, cls in enumerate(STAGE2_CLASSES)}
    y_pattern = df_combined["label"].apply(
        lambda x: label_to_idx.get(x, -1)
    ).values.astype(np.int32)

    # ── Lot groups ───────────────────────────────────────────────────────────
    lot_groups = (
        df_combined["lotName"].values
        if "lotName" in df_combined.columns
        else None
    )

    # Provenance required to prove that demo labels refer to the held-out
    # source records rather than inferred/display-only labels.
    source_indices = df_combined.index.to_numpy(copy=True)
    original_shapes = np.asarray(
        [np.asarray(wafer).shape for wafer in df_combined["waferMap"].values],
        dtype=np.int32,
    )
    raw_failure_types = np.empty(len(df_combined), dtype=object)
    raw_failure_types[:] = [
        np.asarray(value).tolist() for value in df_combined["failureType"].values
    ]

    # ── Save to disk ─────────────────────────────────────────────────────────
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    X_path      = os.path.join(processed_dir, "X.npy")
    labels_path = os.path.join(processed_dir, "labels.npz")

    np.save(X_path, X)
    np.savez(
        labels_path,
        y_binary=y_binary,
        y_pattern=y_pattern,
        lot_groups=lot_groups if lot_groups is not None else np.array([]),
        source_indices=source_indices,
        original_shapes=original_shapes,
        raw_failure_types=raw_failure_types,
    )

    print(f"Saved X         → {X_path}  ({X.nbytes / 1e6:.1f} MB)")
    print(f"Saved labels    → {labels_path}")

    return X, y_binary, y_pattern, lot_groups
