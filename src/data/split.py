"""Global, group-aware train/validation/test splitting for SemVision.

One nine-class partition is created first (``none`` plus eight defect classes).
Stage 1 uses that partition unchanged. Stage 2 is derived by filtering defect
rows inside each global partition. This guarantees that no wafer can be train
data for one stage and held-out test data for the other.
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import (
    RANDOM_SEED,
    STAGE2_CLASSES,
    TEST_SPLIT,
    USE_LOT_GROUPING,
    VAL_SPLIT,
)


GLOBAL_CLASS_NAMES = ["none", *STAGE2_CLASSES]
GROUP_SPLIT_CANDIDATES = 256


def _stratified_train_val_test(indices, labels, test_size, val_size):
    """Return one deterministic three-way stratified split."""
    test_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=RANDOM_SEED
    )
    relative_train_val, relative_test = next(test_splitter.split(indices, labels))
    test_idx = indices[relative_test]
    train_val_idx = indices[relative_train_val]
    train_val_labels = labels[relative_train_val]

    validation_ratio = val_size / (1.0 - test_size)
    validation_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_ratio, random_state=RANDOM_SEED
    )
    relative_train, relative_val = next(
        validation_splitter.split(train_val_idx, train_val_labels)
    )
    return train_val_idx[relative_train], train_val_idx[relative_val], test_idx


def _best_group_partition(indices, labels, groups, holdout_size):
    """Choose a group partition that retains every class on both sides."""
    classes = np.unique(labels)
    splitter = GroupShuffleSplit(
        n_splits=GROUP_SPLIT_CANDIDATES,
        test_size=holdout_size,
        random_state=RANDOM_SEED,
    )
    best = None
    best_score = None
    overall_distribution = np.array([(labels == cls).mean() for cls in classes])

    for relative_train, relative_holdout in splitter.split(indices, labels, groups):
        train_labels = labels[relative_train]
        holdout_labels = labels[relative_holdout]
        if set(np.unique(train_labels)) != set(classes):
            continue
        if set(np.unique(holdout_labels)) != set(classes):
            continue

        holdout_distribution = np.array(
            [(holdout_labels == cls).mean() for cls in classes]
        )
        size_error = abs(len(relative_holdout) / len(indices) - holdout_size)
        distribution_error = float(
            np.mean(np.abs(holdout_distribution - overall_distribution))
        )
        score = size_error + distribution_error
        if best_score is None or score < best_score:
            best_score = score
            best = (relative_train, relative_holdout)

    if best is None:
        raise ValueError(
            f"none of {GROUP_SPLIT_CANDIDATES} grouped candidates preserved "
            f"all {len(classes)} classes in both partitions"
        )
    return best


def _grouped_train_val_test(indices, labels, groups, test_size, val_size):
    """Return one class-covering, lot-disjoint three-way split."""
    relative_train_val, relative_test = _best_group_partition(
        indices, labels, groups, test_size
    )
    train_val_idx = indices[relative_train_val]
    test_idx = indices[relative_test]
    train_val_labels = labels[relative_train_val]
    train_val_groups = groups[relative_train_val]

    validation_ratio = val_size / (1.0 - test_size)
    relative_train, relative_val = _best_group_partition(
        train_val_idx, train_val_labels, train_val_groups, validation_ratio
    )
    return train_val_idx[relative_train], train_val_idx[relative_val], test_idx


def _usable_lot_groups(lot_groups, expected_length):
    """Validate and normalize scalar lot identifiers."""
    if not USE_LOT_GROUPING:
        return False, None, "USE_LOT_GROUPING is disabled", 0
    if lot_groups is None:
        return False, None, "lotName metadata is unavailable", 0

    values = np.asarray(lot_groups, dtype=object)
    if values.ndim != 1 or len(values) != expected_length:
        return False, None, "lotName is not a one-dimensional value per wafer", 0

    normalized = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            return False, None, "lotName contains non-scalar values", 0
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False, None, "lotName contains missing values", 0
        text = str(value).strip()
        if not text:
            return False, None, "lotName contains empty values", 0
        normalized.append(text)

    normalized = np.asarray(normalized, dtype=str)
    unique_count = int(np.unique(normalized).size)
    if unique_count < 20:
        return (
            False,
            None,
            f"lotName has only {unique_count} unique groups; at least 20 are required",
            unique_count,
        )
    return True, normalized, "", unique_count


def _validate_partition(split_indices, expected_indices, groups, name):
    """Prove index exclusivity, coverage, and (when supplied) group isolation."""
    train_idx, val_idx, test_idx = (np.asarray(x, dtype=int) for x in split_indices)
    split_sets = [set(x.tolist()) for x in (train_idx, val_idx, test_idx)]
    assert not split_sets[0].intersection(split_sets[1]), f"{name}: train/val overlap"
    assert not split_sets[0].intersection(split_sets[2]), f"{name}: train/test overlap"
    assert not split_sets[1].intersection(split_sets[2]), f"{name}: val/test overlap"
    assert set.union(*split_sets) == set(np.asarray(expected_indices).tolist()), (
        f"{name}: partitions do not cover every expected sample exactly once"
    )

    if groups is not None:
        group_by_index = dict(
            zip(np.asarray(expected_indices).tolist(), np.asarray(groups).tolist())
        )
        group_sets = [
            {group_by_index[index] for index in split_set}
            for split_set in split_sets
        ]
        assert not group_sets[0].intersection(group_sets[1]), f"{name}: train/val lot overlap"
        assert not group_sets[0].intersection(group_sets[2]), f"{name}: train/test lot overlap"
        assert not group_sets[1].intersection(group_sets[2]), f"{name}: val/test lot overlap"


def _class_counts(labels, indices):
    return {
        name: int(np.sum(labels[np.asarray(indices)] == class_index))
        for class_index, name in enumerate(GLOBAL_CLASS_NAMES)
    }


def _save_dataset_summary(
    labels, global_splits, split_method, fallback_reason, unique_lots, groups
):
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    reports_dir = os.path.join(project_root, "artifacts", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    train_idx, val_idx, test_idx = global_splits
    summary = {
        "total_samples": int(len(labels)),
        "class_names": GLOBAL_CLASS_NAMES,
        "split_method": split_method,
        "grouping_applied": split_method == "lot-grouped",
        "grouping_fallback_reason": fallback_reason or None,
        "unique_lots": int(unique_lots),
        "global_split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "global_class_counts": {
            "all": _class_counts(labels, np.arange(len(labels))),
            "train": _class_counts(labels, train_idx),
            "validation": _class_counts(labels, val_idx),
            "test": _class_counts(labels, test_idx),
        },
        "group_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        } if groups is not None and split_method == "lot-grouped" else None,
    }
    path = os.path.join(reports_dir, "dataset_summary.json")
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    print(f"Dataset summary saved -> {path}")
    return summary


def make_splits(X, y_binary, y_pattern, lot_groups, persist=True):
    """Create one global split and derive both stages from it."""
    if not (len(X) == len(y_binary) == len(y_pattern)):
        raise ValueError("X, y_binary, and y_pattern must have identical lengths")
    all_indices = np.arange(len(y_binary), dtype=int)
    global_labels = np.where(np.asarray(y_pattern) == -1, 0, np.asarray(y_pattern) + 1)
    if set(np.unique(global_labels)) != set(range(len(GLOBAL_CLASS_NAMES))):
        raise ValueError("The dataset must contain none plus all eight defect classes")

    groups_usable, normalized_groups, fallback_reason, unique_lots = _usable_lot_groups(
        lot_groups, len(y_binary)
    )
    split_method = "stratified"
    groups_for_validation = None

    if groups_usable:
        try:
            global_train, global_val, global_test = _grouped_train_val_test(
                all_indices,
                global_labels,
                normalized_groups,
                TEST_SPLIT,
                VAL_SPLIT,
            )
            split_method = "lot-grouped"
            fallback_reason = ""
            groups_for_validation = normalized_groups
        except ValueError as error:
            fallback_reason = f"grouped split rejected: {error}"

    if split_method == "stratified":
        global_train, global_val, global_test = _stratified_train_val_test(
            all_indices, global_labels, TEST_SPLIT, VAL_SPLIT
        )
        print(
            "WARNING: using a stratified sample-level split. Lot isolation is not "
            f"guaranteed. Reason: {fallback_reason}"
        )

    global_splits = (global_train, global_val, global_test)
    _validate_partition(
        global_splits, all_indices, groups_for_validation, "Global split"
    )

    stage2_splits = tuple(
        partition[np.asarray(y_pattern)[partition] != -1]
        for partition in global_splits
    )
    stage1_splits = tuple(partition.copy() for partition in global_splits)
    defect_indices = np.flatnonzero(np.asarray(y_pattern) != -1)
    _validate_partition(stage1_splits, all_indices, groups_for_validation, "Stage 1")
    _validate_partition(
        stage2_splits,
        defect_indices,
        normalized_groups[defect_indices] if split_method == "lot-grouped" else None,
        "Stage 2",
    )
    for global_partition, stage1_partition, stage2_partition in zip(
        global_splits, stage1_splits, stage2_splits
    ):
        assert np.array_equal(global_partition, stage1_partition)
        assert np.array_equal(
            stage2_partition,
            global_partition[np.asarray(y_pattern)[global_partition] != -1],
        )

    splits_dict = {
        "global_train_idx": global_train,
        "global_val_idx": global_val,
        "global_test_idx": global_test,
        "stage1_train_idx": stage1_splits[0],
        "stage1_val_idx": stage1_splits[1],
        "stage1_test_idx": stage1_splits[2],
        "stage2_train_idx": stage2_splits[0],
        "stage2_val_idx": stage2_splits[1],
        "stage2_test_idx": stage2_splits[2],
        "split_method": split_method,
        "split_fallback_reason": fallback_reason,
    }

    if persist:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        processed_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        np.savez(os.path.join(processed_dir, "splits.npz"), **splits_dict)
        _save_dataset_summary(
            global_labels,
            global_splits,
            split_method,
            fallback_reason,
            unique_lots,
            groups_for_validation,
        )

    _print_split_summary(splits_dict, global_labels)
    return splits_dict


def _print_split_summary(splits_dict, global_labels):
    print("\nGLOBAL SPLIT SUMMARY")
    for split_name, key in (
        ("Train", "global_train_idx"),
        ("Validation", "global_val_idx"),
        ("Test", "global_test_idx"),
    ):
        indices = splits_dict[key]
        counts = " ".join(
            f"{name}={count}"
            for name, count in _class_counts(global_labels, indices).items()
        )
        print(f"  {split_name:<10} n={len(indices):>6,}: {counts}")
    print(
        "  Derived Stage 2 sizes: "
        f"train={len(splits_dict['stage2_train_idx']):,}, "
        f"validation={len(splits_dict['stage2_val_idx']):,}, "
        f"test={len(splits_dict['stage2_test_idx']):,}"
    )
