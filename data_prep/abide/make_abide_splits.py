#!/usr/bin/env python3

import argparse
import os
import re
import json

import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


SEEDS = [42, 1042, 2042, 3042, 4042]


def numeric_subject_id(value):
    """
    Convert subject IDs such as:
        sub-control50002
        sub-patient50003
        50002
    -> integer ID
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    nums = re.findall(r"\d+", str(value))

    if not nums:
        raise ValueError(f"Cannot extract numeric ID from {value}")

    return int(nums[-1])


def load_old_static_cohort(static_h5):
    """
    Load subjects exactly the same way as phase1_baseline_fc.py.

    Returns
    -------
    subject_ids : (N,)
    labels      : (N,)
    """
    subject_ids = []
    labels = []

    with h5py.File(static_h5, "r") as f:

        for subject_key in f.keys():

            if "schaefer100" not in f[subject_key]:
                continue

            sid = numeric_subject_id(subject_key)

            label = int(
                f[subject_key]["schaefer100"].attrs["label"]
            )

            subject_ids.append(sid)
            labels.append(label)

    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    if len(subject_ids) == 0:
        raise RuntimeError("No Schaefer100 subjects found.")

    if len(np.unique(subject_ids)) != len(subject_ids):
        raise RuntimeError("Duplicate subject IDs in old static HDF5.")

    return subject_ids, labels


def load_temporal_cohort(temporal_h5):
    """
    Read IDs and labels from your Step-2 temporal HDF5.
    """
    with h5py.File(temporal_h5, "r") as f:
        subject_ids = f["subject_id"][:].astype(np.int64)
        labels = f["label"][:].astype(np.int64)

    if len(np.unique(subject_ids)) != len(subject_ids):
        raise RuntimeError("Duplicate subject IDs in temporal HDF5.")

    return subject_ids, labels


def verify_cohorts(
    static_ids,
    static_labels,
    temporal_ids,
    temporal_labels,
):
    """
    Verify same subjects and same diagnosis labels,
    independent of subject ordering.
    """

    static_map = {
        int(sid): int(label)
        for sid, label in zip(static_ids, static_labels)
    }

    temporal_map = {
        int(sid): int(label)
        for sid, label in zip(temporal_ids, temporal_labels)
    }

    static_set = set(static_map)
    temporal_set = set(temporal_map)

    only_static = sorted(static_set - temporal_set)
    only_temporal = sorted(temporal_set - static_set)

    if only_static:
        raise RuntimeError(
            f"{len(only_static)} subjects only in static dataset. "
            f"Examples: {only_static[:20]}"
        )

    if only_temporal:
        raise RuntimeError(
            f"{len(only_temporal)} subjects only in temporal dataset. "
            f"Examples: {only_temporal[:20]}"
        )

    mismatches = []

    for sid in static_set:
        if static_map[sid] != temporal_map[sid]:
            mismatches.append(
                (sid, static_map[sid], temporal_map[sid])
            )

    if mismatches:
        raise RuntimeError(
            f"{len(mismatches)} label mismatches. "
            f"Examples: {mismatches[:20]}"
        )

    print("Static vs temporal subjects: OK")
    print("Static vs temporal labels:   OK")


def counts(labels):
    return {
        "control": int(np.sum(labels == 0)),
        "patient": int(np.sum(labels == 1)),
    }


def make_split(subject_ids, labels, seed):
    """
    EXACT logic from your old phase1_baseline_fc.py.

    First:
        80% train+val
        20% test

    Then:
        12.5% of the 80% -> validation

    Final:
        70% train
        10% val
        20% test
    """

    # 80 / 20
    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_val_idx, test_idx = next(
        sss_test.split(subject_ids, labels)
    )

    # 70 / 10 from remaining 80
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=seed,
    )

    train_rel_idx, val_rel_idx = next(
        sss_val.split(
            subject_ids[train_val_idx],
            labels[train_val_idx],
        )
    )

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    return train_idx, val_idx, test_idx


def verify_split(train_ids, val_ids, test_ids, all_ids):
    train_set = set(train_ids.tolist())
    val_set = set(val_ids.tolist())
    test_set = set(test_ids.tolist())

    if not train_set.isdisjoint(val_set):
        raise RuntimeError("Train/val overlap found.")

    if not train_set.isdisjoint(test_set):
        raise RuntimeError("Train/test overlap found.")

    if not val_set.isdisjoint(test_set):
        raise RuntimeError("Val/test overlap found.")

    combined = train_set | val_set | test_set

    if combined != set(all_ids.tolist()):
        raise RuntimeError("Split does not cover complete cohort.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--static-h5",
        required=True,
        help="Old abide_data_tr.h5 used in previous project",
    )

    parser.add_argument(
        "--temporal-h5",
        required=True,
        help="New temporal FC HDF5",
    )

    parser.add_argument(
        "--out-dir",
        default="splits",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading old static cohort...")

    static_ids, static_labels = load_old_static_cohort(
        args.static_h5
    )

    print(
        f"Static: {len(static_ids)} subjects "
        f"{counts(static_labels)}"
    )

    print("\nLoading temporal cohort...")

    temporal_ids, temporal_labels = load_temporal_cohort(
        args.temporal_h5
    )

    print(
        f"Temporal: {len(temporal_ids)} subjects "
        f"{counts(temporal_labels)}"
    )

    print("\nVerifying cohorts...")

    verify_cohorts(
        static_ids,
        static_labels,
        temporal_ids,
        temporal_labels,
    )

    summary = {
        "protocol": "stratified 70/10/20",
        "seeds": SEEDS,
        "n_subjects": int(len(static_ids)),
        "class_distribution": counts(static_labels),
        "splits": {},
    }

    print("\nCreating frozen splits...")

    for seed in SEEDS:

        train_idx, val_idx, test_idx = make_split(
            static_ids,
            static_labels,
            seed,
        )

        # IMPORTANT:
        # convert the old array indices into SUBJECT IDs
        train_ids = static_ids[train_idx]
        val_ids = static_ids[val_idx]
        test_ids = static_ids[test_idx]

        train_labels = static_labels[train_idx]
        val_labels = static_labels[val_idx]
        test_labels = static_labels[test_idx]

        verify_split(
            train_ids,
            val_ids,
            test_ids,
            static_ids,
        )

        out_file = os.path.join(
            args.out_dir,
            f"abide_split_seed{seed}.npz",
        )

        np.savez_compressed(
            out_file,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            seed=np.array(seed, dtype=np.int64),
        )

        summary["splits"][str(seed)] = {
            "train": {
                "n": int(len(train_ids)),
                **counts(train_labels),
            },
            "val": {
                "n": int(len(val_ids)),
                **counts(val_labels),
            },
            "test": {
                "n": int(len(test_ids)),
                **counts(test_labels),
            },
        }

        print(f"\nSeed {seed}")

        print(
            f"  Train: {len(train_ids):4d} "
            f"{counts(train_labels)}"
        )

        print(
            f"  Val:   {len(val_ids):4d} "
            f"{counts(val_labels)}"
        )

        print(
            f"  Test:  {len(test_ids):4d} "
            f"{counts(test_labels)}"
        )

        print(f"  saved -> {out_file}")

    summary_file = os.path.join(
        args.out_dir,
        "split_summary.json",
    )

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========================================")
    print("ALL SPLITS CREATED SUCCESSFULLY")
    print("========================================")
    print(f"Summary -> {summary_file}")


if __name__ == "__main__":
    main()