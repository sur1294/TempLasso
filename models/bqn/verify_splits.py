#!/usr/bin/env python3
"""
Verify frozen ABIDE Schaefer100 splits against the OLD static HDF5.

This script:
- uses ONLY derivatives/abide_data_tr.h5
- uses the existing frozen NPZ split files in splits/
- matches subjects by numeric SUBJECT ID
- does NOT generate any new split
- does NOT use temporal data
- does NOT fit any preprocessing/statistics

Expected split files:
    splits/abide_split_seed42.npz
    splits/abide_split_seed1042.npz
    splits/abide_split_seed2042.npz
    splits/abide_split_seed3042.npz
    splits/abide_split_seed4042.npz

Each NPZ must contain:
    train_ids, val_ids, test_ids,
    train_labels, val_labels, test_labels
"""

import argparse
import os
import re

import h5py
import numpy as np


SEEDS = [42, 1042, 2042, 3042, 4042]
ATLAS = "schaefer100"

EXPECTED_N = 1025
EXPECTED_COUNTS = {
    "train": 717,
    "val": 103,
    "test": 205,
}
EXPECTED_CLASS_COUNTS = {
    "train": {0: 376, 1: 341},
    "val":   {0: 54,  1: 49},
    "test":  {0: 107, 1: 98},
}


def numeric_subject_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    nums = re.findall(r"\d+", str(value))
    if not nums:
        raise ValueError(f"Cannot extract numeric subject ID from: {value}")

    return int(nums[-1])


def load_static_subject_map(h5_path):
    """
    Returns
    -------
    static_map : dict[int, dict]
        sid -> {
            "label": int,
            "subject_key": str
        }
    """
    static_map = {}

    with h5py.File(h5_path, "r") as f:
        for subject_key in f.keys():
            if ATLAS not in f[subject_key]:
                continue

            grp = f[subject_key][ATLAS]

            if "corr_matrix" not in grp:
                raise RuntimeError(
                    f"{subject_key}/{ATLAS} is missing corr_matrix"
                )

            sid = numeric_subject_id(subject_key)
            label = int(grp.attrs["label"])

            if sid in static_map:
                raise RuntimeError(f"Duplicate numeric subject ID in HDF5: {sid}")

            static_map[sid] = {
                "label": label,
                "subject_key": str(subject_key),
            }

    return static_map


def label_counts(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return {
        0: int(np.sum(labels == 0)),
        1: int(np.sum(labels == 1)),
    }


def check_one_split(seed, split_path, static_map):
    data = np.load(split_path, allow_pickle=False)

    required = {
        "train_ids",
        "val_ids",
        "test_ids",
        "train_labels",
        "val_labels",
        "test_labels",
    }

    missing_keys = required - set(data.files)
    if missing_keys:
        raise RuntimeError(
            f"Seed {seed}: split file missing keys: {sorted(missing_keys)}"
        )

    ids = {
        "train": np.asarray(data["train_ids"], dtype=np.int64),
        "val": np.asarray(data["val_ids"], dtype=np.int64),
        "test": np.asarray(data["test_ids"], dtype=np.int64),
    }

    labels = {
        "train": np.asarray(data["train_labels"], dtype=np.int64),
        "val": np.asarray(data["val_labels"], dtype=np.int64),
        "test": np.asarray(data["test_labels"], dtype=np.int64),
    }

    print(f"\nSeed {seed}")
    print("-" * 72)
    print(f"Split file: {split_path}")

    # ------------------------------------------------------------
    # 1. Count checks
    # ------------------------------------------------------------
    for split_name in ("train", "val", "test"):
        n = len(ids[split_name])

        if n != EXPECTED_COUNTS[split_name]:
            raise RuntimeError(
                f"Seed {seed} {split_name}: expected "
                f"{EXPECTED_COUNTS[split_name]} IDs, got {n}"
            )

        if len(labels[split_name]) != n:
            raise RuntimeError(
                f"Seed {seed} {split_name}: ID/label length mismatch "
                f"{n} vs {len(labels[split_name])}"
            )

        counts = label_counts(labels[split_name])

        if counts != EXPECTED_CLASS_COUNTS[split_name]:
            raise RuntimeError(
                f"Seed {seed} {split_name}: expected class counts "
                f"{EXPECTED_CLASS_COUNTS[split_name]}, got {counts}"
            )

        print(
            f"  {split_name:5s}: n={n:3d} "
            f"control={counts[0]:3d} patient={counts[1]:3d}"
        )

    # ------------------------------------------------------------
    # 2. Duplicate checks inside each split
    # ------------------------------------------------------------
    for split_name in ("train", "val", "test"):
        arr = ids[split_name]

        if len(np.unique(arr)) != len(arr):
            raise RuntimeError(
                f"Seed {seed} {split_name}: duplicate subject IDs found"
            )

    # ------------------------------------------------------------
    # 3. Disjointness
    # ------------------------------------------------------------
    train_set = set(ids["train"].tolist())
    val_set = set(ids["val"].tolist())
    test_set = set(ids["test"].tolist())

    if not train_set.isdisjoint(val_set):
        raise RuntimeError(f"Seed {seed}: train/val overlap found")

    if not train_set.isdisjoint(test_set):
        raise RuntimeError(f"Seed {seed}: train/test overlap found")

    if not val_set.isdisjoint(test_set):
        raise RuntimeError(f"Seed {seed}: val/test overlap found")

    print("  disjoint train/val/test: PASS")

    # ------------------------------------------------------------
    # 4. Union must be exactly the full static cohort
    # ------------------------------------------------------------
    union = train_set | val_set | test_set
    static_set = set(static_map.keys())

    if len(union) != EXPECTED_N:
        raise RuntimeError(
            f"Seed {seed}: union has {len(union)} subjects, expected {EXPECTED_N}"
        )

    missing_from_split = sorted(static_set - union)
    extra_in_split = sorted(union - static_set)

    if missing_from_split:
        raise RuntimeError(
            f"Seed {seed}: {len(missing_from_split)} static subjects missing "
            f"from split. Examples: {missing_from_split[:20]}"
        )

    if extra_in_split:
        raise RuntimeError(
            f"Seed {seed}: {len(extra_in_split)} split subjects not in "
            f"static HDF5. Examples: {extra_in_split[:20]}"
        )

    print(f"  union == all {EXPECTED_N} static subjects: PASS")

    # ------------------------------------------------------------
    # 5. Every split ID must exist in old static HDF5
    #    and every split label must match old static HDF5
    # ------------------------------------------------------------
    mismatch_rows = []

    for split_name in ("train", "val", "test"):
        for sid, split_label in zip(
            ids[split_name].tolist(),
            labels[split_name].tolist(),
        ):
            if sid not in static_map:
                raise RuntimeError(
                    f"Seed {seed} {split_name}: subject {sid} not found "
                    f"in old static HDF5"
                )

            h5_label = int(static_map[sid]["label"])

            if int(split_label) != h5_label:
                mismatch_rows.append(
                    (
                        split_name,
                        int(sid),
                        int(split_label),
                        h5_label,
                    )
                )

    if mismatch_rows:
        raise RuntimeError(
            f"Seed {seed}: {len(mismatch_rows)} split/HDF5 label mismatches. "
            f"Examples: {mismatch_rows[:20]}"
        )

    print("  all split IDs exist in old static HDF5: PASS")
    print("  all split labels match old static HDF5: PASS")

    # ------------------------------------------------------------
    # 6. Seed metadata, if present
    # ------------------------------------------------------------
    if "seed" in data.files:
        stored_seed = int(np.asarray(data["seed"]).item())
        if stored_seed != seed:
            raise RuntimeError(
                f"Seed metadata mismatch: filename seed={seed}, "
                f"stored seed={stored_seed}"
            )
        print(f"  stored seed == {seed}: PASS")

    print("  RESULT: PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Verify frozen ABIDE split files against old static HDF5."
    )

    parser.add_argument(
        "--h5",
        default="derivatives/abide_data_tr.h5",
        help="Old static ABIDE HDF5.",
    )

    parser.add_argument(
        "--split-dir",
        default="splits",
        help="Directory containing abide_split_seed*.npz",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("VERIFY FROZEN STATIC BQN SPLITS")
    print("=" * 72)
    print(f"Static HDF5: {args.h5}")
    print(f"Split dir:   {args.split_dir}")

    static_map = load_static_subject_map(args.h5)

    static_labels = np.asarray(
        [v["label"] for v in static_map.values()],
        dtype=np.int64,
    )

    print(f"\nStatic Schaefer100 subjects: {len(static_map)}")
    print(f"Static label counts: {label_counts(static_labels)}")

    if len(static_map) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} static subjects, found {len(static_map)}"
        )

    if label_counts(static_labels) != {0: 537, 1: 488}:
        raise RuntimeError(
            "Static HDF5 class distribution does not match expected "
            "{0: 537, 1: 488}"
        )

    for seed in SEEDS:
        split_path = os.path.join(
            args.split_dir,
            f"abide_split_seed{seed}.npz",
        )

        if not os.path.exists(split_path):
            raise FileNotFoundError(split_path)

        check_one_split(
            seed=seed,
            split_path=split_path,
            static_map=static_map,
        )

    print("\n" + "=" * 72)
    print("ALL FIVE FROZEN SPLITS VERIFIED SUCCESSFULLY")
    print("=" * 72)
    print(
        "\nNo random split was generated.\n"
        "No temporal HDF5 was used.\n"
        "No preprocessing or statistics were fitted.\n"
        "Subject selection is verified by subject ID."
    )


if __name__ == "__main__":
    main()
