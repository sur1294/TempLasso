#!/usr/bin/env python3
"""
Inspect the OLD static ABIDE HDF5 used for the Schaefer100 BQN baseline.

This script is inspection-only:
- does NOT modify the HDF5
- does NOT create train/val/test splits
- does NOT use the temporal HDF5
- does NOT apply Fisher-z, scaling, thresholding, normalization, or any other preprocessing

Expected static HDF5 structure:
    /<subject_key>/schaefer100/corr_matrix
    /<subject_key>/schaefer100.attrs["label"]

Run from the Brainnetwork project root:
    python models/bqn/inspect_static_data.py \
        --h5 derivatives/abide_data_tr.h5

The goal is to establish exactly what the static BQN receives.
"""

import argparse
import re
from collections import Counter

import h5py
import numpy as np


ATLAS = "schaefer100"
EXPECTED_ROIS = 100
EXPECTED_SUBJECTS = 1025
EXPECTED_LABEL_COUNTS = {0: 537, 1: 488}


def numeric_subject_id(value):
    """
    Convert IDs such as:
        sub-control50002
        sub-patient50003
        50002
    into integer subject IDs.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    nums = re.findall(r"\d+", str(value))
    if not nums:
        raise ValueError(f"Cannot extract numeric subject ID from: {value}")

    return int(nums[-1])


def offdiag_values(matrix):
    """Return all off-diagonal entries."""
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect old static ABIDE Schaefer100 FC data."
    )
    parser.add_argument(
        "--h5",
        default="derivatives/abide_data_tr.h5",
        help="Path to old static ABIDE HDF5.",
    )
    parser.add_argument(
        "--atlas",
        default=ATLAS,
        help="Atlas group to inspect.",
    )
    parser.add_argument(
        "--show-subjects",
        type=int,
        default=5,
        help="How many example subject keys/IDs to print.",
    )
    parser.add_argument(
        "--symmetry-tol",
        type=float,
        default=1e-6,
        help="Tolerance for symmetry checks.",
    )
    args = parser.parse_args()

    subject_keys = []
    subject_ids = []
    labels = []

    min_vals = []
    max_vals = []
    means = []
    stds = []
    diag_means = []
    diag_mins = []
    diag_maxs = []
    symmetry_errors = []
    negative_fracs = []
    zero_fracs = []

    nonfinite_subjects = []
    wrong_shape_subjects = []
    asymmetric_subjects = []

    global_min = np.inf
    global_max = -np.inf

    print("=" * 72)
    print("STATIC ABIDE / SCHAEFER100 INSPECTION")
    print("=" * 72)
    print(f"HDF5:  {args.h5}")
    print(f"Atlas: {args.atlas}")
    print()

    with h5py.File(args.h5, "r") as f:
        root_keys = list(f.keys())

        print(f"Top-level HDF5 groups: {len(root_keys)}")

        for subject_key in root_keys:
            if args.atlas not in f[subject_key]:
                continue

            grp = f[subject_key][args.atlas]

            if "corr_matrix" not in grp:
                raise RuntimeError(
                    f"{subject_key}/{args.atlas} has no corr_matrix dataset."
                )

            corr = np.asarray(grp["corr_matrix"][:], dtype=np.float64)

            if "label" not in grp.attrs:
                raise RuntimeError(
                    f"{subject_key}/{args.atlas} has no label attribute."
                )

            label = int(grp.attrs["label"])
            sid = numeric_subject_id(subject_key)

            subject_keys.append(str(subject_key))
            subject_ids.append(sid)
            labels.append(label)

            if corr.shape != (EXPECTED_ROIS, EXPECTED_ROIS):
                wrong_shape_subjects.append((subject_key, corr.shape))
                continue

            if not np.isfinite(corr).all():
                nonfinite_subjects.append(subject_key)
                continue

            sym_err = float(np.max(np.abs(corr - corr.T)))
            symmetry_errors.append(sym_err)

            if sym_err > args.symmetry_tol:
                asymmetric_subjects.append((subject_key, sym_err))

            diag = np.diag(corr)
            offdiag = offdiag_values(corr)

            min_vals.append(float(corr.min()))
            max_vals.append(float(corr.max()))
            means.append(float(corr.mean()))
            stds.append(float(corr.std()))

            diag_means.append(float(diag.mean()))
            diag_mins.append(float(diag.min()))
            diag_maxs.append(float(diag.max()))

            negative_fracs.append(float(np.mean(offdiag < 0)))
            zero_fracs.append(float(np.mean(offdiag == 0)))

            global_min = min(global_min, float(corr.min()))
            global_max = max(global_max, float(corr.max()))

    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    print()
    print("=== COHORT ===")
    print(f"Schaefer100 subjects: {len(subject_ids)}")

    label_counts = dict(
        zip(*np.unique(labels, return_counts=True))
    )
    label_counts = {int(k): int(v) for k, v in label_counts.items()}

    print(f"Label counts: {label_counts}")
    print(f"Unique subject IDs: {len(np.unique(subject_ids))}")
    print(f"Duplicate subject IDs: {len(subject_ids) - len(np.unique(subject_ids))}")

    if len(subject_ids):
        nshow = min(args.show_subjects, len(subject_ids))
        print("\nExample subjects:")
        for key, sid, label in zip(
            subject_keys[:nshow],
            subject_ids[:nshow],
            labels[:nshow],
        ):
            print(f"  key={key}   numeric_id={sid}   label={label}")

    print()
    print("=== EXPECTED COHORT CHECK ===")
    print(
        f"N == {EXPECTED_SUBJECTS}: "
        f"{'PASS' if len(subject_ids) == EXPECTED_SUBJECTS else 'FAIL'}"
    )
    print(
        f"labels == {EXPECTED_LABEL_COUNTS}: "
        f"{'PASS' if label_counts == EXPECTED_LABEL_COUNTS else 'FAIL'}"
    )

    print()
    print("=== MATRIX VALIDITY ===")
    print(f"Wrong-shape subjects: {len(wrong_shape_subjects)}")
    if wrong_shape_subjects:
        print("  examples:", wrong_shape_subjects[:10])

    print(f"Non-finite subjects: {len(nonfinite_subjects)}")
    if nonfinite_subjects:
        print("  examples:", nonfinite_subjects[:10])

    print(
        f"Asymmetric subjects (tol={args.symmetry_tol:g}): "
        f"{len(asymmetric_subjects)}"
    )
    if asymmetric_subjects:
        print("  examples:", asymmetric_subjects[:10])

    if symmetry_errors:
        print(
            "Maximum symmetry error across subjects: "
            f"{max(symmetry_errors):.3e}"
        )

    valid_count = len(min_vals)

    if valid_count == 0:
        raise RuntimeError("No valid Schaefer100 correlation matrices found.")

    print()
    print("=== FC VALUE DISTRIBUTION ===")
    print(f"Valid matrices summarized: {valid_count}")
    print(f"Global minimum: {global_min:.8f}")
    print(f"Global maximum: {global_max:.8f}")
    print(f"Mean of subject matrix means: {np.mean(means):.8f}")
    print(f"Mean of subject matrix stds:  {np.mean(stds):.8f}")

    print()
    print("=== DIAGONAL ===")
    print(f"Mean diagonal value: {np.mean(diag_means):.8f}")
    print(f"Minimum diagonal value observed: {np.min(diag_mins):.8f}")
    print(f"Maximum diagonal value observed: {np.max(diag_maxs):.8f}")

    print()
    print("=== OFF-DIAGONAL SIGN / ZERO STRUCTURE ===")
    print(
        "Mean fraction negative off-diagonal entries: "
        f"{np.mean(negative_fracs):.6f}"
    )
    print(
        "Mean fraction exactly-zero off-diagonal entries: "
        f"{np.mean(zero_fracs):.6f}"
    )

    print()
    print("=== REPRESENTATION DIAGNOSTIC ===")

    if global_min >= -1.000001 and global_max <= 1.000001:
        print(
            "All observed values are within approximately [-1, 1].\n"
            "This is consistent with a correlation-scale representation."
        )
    else:
        print(
            "Observed values extend outside [-1, 1].\n"
            "Therefore the stored matrix is NOT plain Pearson-r scale.\n"
            "Possible explanations include Fisher-z or another transform.\n"
            "Do not change the representation until its provenance is verified."
        )

    if np.mean(negative_fracs) > 0:
        print("Negative FC values are present: the stored FC is signed.")
    else:
        print(
            "No negative off-diagonal FC values were observed. "
            "Check whether correlations were thresholded/clipped."
        )

    if np.mean(zero_fracs) > 0.01:
        print(
            "A noticeable fraction of off-diagonal values are exactly zero. "
            "This may indicate thresholding/sparsification and should be verified."
        )
    else:
        print(
            "Very few/no exact-zero off-diagonal entries were observed; "
            "the matrices appear dense."
        )

    print()
    print("=== WHAT STATIC BQN RECEIVES ===")
    print(
        "This inspection script performs NO transformation.\n"
        "Therefore, if the training loader reads corr_matrix directly, "
        "the static BQN input per subject is exactly the stored "
        f"{EXPECTED_ROIS}x{EXPECTED_ROIS} Schaefer100 matrix."
    )

    print()
    print("=" * 72)
    print("INSPECTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
