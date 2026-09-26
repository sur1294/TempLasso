#!/usr/bin/env python3
"""
Static ABIDE Schaefer100 dataset utilities for BQN.

This file contains DATA LOADING ONLY.

It:
- reads the OLD static file: derivatives/abide_data_tr.h5
- reads one existing frozen split NPZ
- matches split subjects to the HDF5 by SUBJECT ID
- returns the stored Schaefer100 corr_matrix exactly as-is
- does NOT use the temporal HDF5
- does NOT generate new splits
- does NOT apply Fisher-z, scaling, normalization, thresholding, or feature fitting

Expected HDF5 structure
-----------------------
/<subject_key>/schaefer100/corr_matrix
/<subject_key>/schaefer100.attrs["label"]

Expected split NPZ keys
-----------------------
train_ids
val_ids
test_ids
train_labels
val_labels
test_labels
seed

Example
-------
python models/bqn/bqn_dataset.py \
    --h5 derivatives/abide_data_tr.h5 \
    --split splits/abide_split_seed42.npz
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ATLAS = "schaefer100"
EXPECTED_N_ROIS = 100
EXPECTED_N_SUBJECTS = 1025
EXPECTED_SPLIT_SIZES = {
    "train": 717,
    "val": 103,
    "test": 205,
}


def numeric_subject_id(value) -> int:
    """
    Convert identifiers such as:
        sub-control50030
        sub-patient50003
        50030
    into integer ABIDE subject ID.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    nums = re.findall(r"\d+", str(value))
    if not nums:
        raise ValueError(f"Cannot extract numeric subject ID from {value!r}")

    return int(nums[-1])


@dataclass(frozen=True)
class StaticSubjectRecord:
    subject_id: int
    subject_key: str
    label: int


class StaticABIDESchaeferDataset(Dataset):
    """
    Dataset for a fixed list of ABIDE subject IDs.

    Each item returns:
        fc         : float32 tensor [100,100]
        label      : long scalar tensor
        subject_id : long scalar tensor

    The FC matrix is read from corr_matrix and returned unchanged except
    for dtype conversion to float32.
    """

    def __init__(
        self,
        h5_path: str,
        subject_ids: np.ndarray,
        expected_labels: np.ndarray,
    ) -> None:
        super().__init__()

        self.h5_path = str(h5_path)
        self.subject_ids = np.asarray(subject_ids, dtype=np.int64)
        self.expected_labels = np.asarray(expected_labels, dtype=np.int64)

        if self.subject_ids.ndim != 1:
            raise ValueError("subject_ids must be 1-D.")
        if self.expected_labels.ndim != 1:
            raise ValueError("expected_labels must be 1-D.")
        if len(self.subject_ids) != len(self.expected_labels):
            raise ValueError(
                "subject_ids and expected_labels must have the same length."
            )
        if len(np.unique(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("Duplicate subject IDs supplied to dataset.")

        self._records = self._build_records()

    def _build_records(self) -> List[StaticSubjectRecord]:
        """
        Resolve every requested subject ID to its HDF5 subject key.

        This is done explicitly by SUBJECT ID, never by HDF5 iteration position.
        """
        id_to_record: Dict[int, StaticSubjectRecord] = {}

        with h5py.File(self.h5_path, "r") as f:
            for subject_key in f.keys():
                if ATLAS not in f[subject_key]:
                    continue

                grp = f[subject_key][ATLAS]

                if "corr_matrix" not in grp:
                    raise RuntimeError(
                        f"{subject_key}/{ATLAS} missing corr_matrix."
                    )
                if "label" not in grp.attrs:
                    raise RuntimeError(
                        f"{subject_key}/{ATLAS} missing label attribute."
                    )

                sid = numeric_subject_id(subject_key)
                label = int(grp.attrs["label"])

                if sid in id_to_record:
                    raise RuntimeError(
                        f"Duplicate numeric subject ID {sid} in static HDF5."
                    )

                id_to_record[sid] = StaticSubjectRecord(
                    subject_id=sid,
                    subject_key=str(subject_key),
                    label=label,
                )

        records: List[StaticSubjectRecord] = []

        for sid, expected_label in zip(
            self.subject_ids.tolist(),
            self.expected_labels.tolist(),
        ):
            if sid not in id_to_record:
                raise KeyError(
                    f"Subject ID {sid} from split file is missing from "
                    f"{self.h5_path}"
                )

            rec = id_to_record[sid]

            if rec.label != int(expected_label):
                raise ValueError(
                    f"Label mismatch for subject {sid}: "
                    f"split={int(expected_label)}, h5={rec.label}"
                )

            records.append(rec)

        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        rec = self._records[index]

        # Open the file inside __getitem__ so the Dataset remains safe with
        # DataLoader workers and does not keep a cross-process h5py handle.
        with h5py.File(self.h5_path, "r") as f:
            grp = f[rec.subject_key][ATLAS]
            fc = np.asarray(grp["corr_matrix"][:], dtype=np.float32)
            label = int(grp.attrs["label"])

        if fc.shape != (EXPECTED_N_ROIS, EXPECTED_N_ROIS):
            raise RuntimeError(
                f"{rec.subject_key}: expected FC shape "
                f"({EXPECTED_N_ROIS},{EXPECTED_N_ROIS}), got {fc.shape}"
            )

        if not np.isfinite(fc).all():
            raise RuntimeError(
                f"{rec.subject_key}: corr_matrix contains NaN/Inf"
            )

        if label != rec.label:
            raise RuntimeError(
                f"{rec.subject_key}: label changed unexpectedly "
                f"({label} != {rec.label})"
            )

        return (
            torch.from_numpy(fc),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(rec.subject_id, dtype=torch.long),
        )


def load_frozen_split(split_path: str) -> Dict[str, np.ndarray]:
    """
    Load one frozen split file.

    No new random split is generated.
    """
    data = np.load(split_path, allow_pickle=False)

    required = {
        "train_ids",
        "val_ids",
        "test_ids",
        "train_labels",
        "val_labels",
        "test_labels",
    }

    missing = required - set(data.files)
    if missing:
        raise RuntimeError(
            f"Split file {split_path} is missing keys: {sorted(missing)}"
        )

    split = {
        "train_ids": np.asarray(data["train_ids"], dtype=np.int64),
        "val_ids": np.asarray(data["val_ids"], dtype=np.int64),
        "test_ids": np.asarray(data["test_ids"], dtype=np.int64),
        "train_labels": np.asarray(data["train_labels"], dtype=np.int64),
        "val_labels": np.asarray(data["val_labels"], dtype=np.int64),
        "test_labels": np.asarray(data["test_labels"], dtype=np.int64),
    }

    if "seed" in data.files:
        split["seed"] = np.asarray(data["seed"], dtype=np.int64)

    return split


def build_datasets(
    h5_path: str,
    split_path: str,
) -> Tuple[
    StaticABIDESchaeferDataset,
    StaticABIDESchaeferDataset,
    StaticABIDESchaeferDataset,
]:
    """
    Build train/val/test datasets from one frozen split.
    """
    split = load_frozen_split(split_path)

    train_ds = StaticABIDESchaeferDataset(
        h5_path=h5_path,
        subject_ids=split["train_ids"],
        expected_labels=split["train_labels"],
    )

    val_ds = StaticABIDESchaeferDataset(
        h5_path=h5_path,
        subject_ids=split["val_ids"],
        expected_labels=split["val_labels"],
    )

    test_ds = StaticABIDESchaeferDataset(
        h5_path=h5_path,
        subject_ids=split["test_ids"],
        expected_labels=split["test_labels"],
    )

    return train_ds, val_ds, test_ds


def build_loaders(
    h5_path: str,
    split_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_seed: int = 42,
):
    """
    Build DataLoaders for one frozen split.

    Only the training loader shuffles. Validation and test preserve split order.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")

    train_ds, val_ds, test_ds = build_datasets(
        h5_path=h5_path,
        split_path=split_path,
    )

    generator = torch.Generator()
    generator.manual_seed(int(shuffle_seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def _class_counts(labels: np.ndarray) -> Dict[int, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        0: int(np.sum(labels == 0)),
        1: int(np.sum(labels == 1)),
    }


def _self_test(
    h5_path: str,
    split_path: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """
    End-to-end dataset sanity check for one frozen split.
    """
    split = load_frozen_split(split_path)

    train_ds, val_ds, test_ds = build_datasets(
        h5_path=h5_path,
        split_path=split_path,
    )

    assert len(train_ds) == EXPECTED_SPLIT_SIZES["train"]
    assert len(val_ds) == EXPECTED_SPLIT_SIZES["val"]
    assert len(test_ds) == EXPECTED_SPLIT_SIZES["test"]

    train_ids = set(split["train_ids"].tolist())
    val_ids = set(split["val_ids"].tolist())
    test_ids = set(split["test_ids"].tolist())

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == EXPECTED_N_SUBJECTS

    train_loader, val_loader, test_loader = build_loaders(
        h5_path=h5_path,
        split_path=split_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_seed=int(
            np.asarray(split.get("seed", np.array(42))).item()
        ),
    )

    fc, y, sid = next(iter(train_loader))

    assert fc.ndim == 3
    assert fc.shape[1:] == (EXPECTED_N_ROIS, EXPECTED_N_ROIS)
    assert y.ndim == 1
    assert sid.ndim == 1
    assert fc.dtype == torch.float32
    assert y.dtype == torch.long
    assert sid.dtype == torch.long
    assert torch.isfinite(fc).all()

    print("StaticABIDESchaeferDataset self-test: PASS")
    print(f"split file: {split_path}")

    if "seed" in split:
        print(f"stored seed: {int(split['seed'].item())}")

    print(
        "train:",
        len(train_ds),
        _class_counts(split["train_labels"]),
    )
    print(
        "val:  ",
        len(val_ds),
        _class_counts(split["val_labels"]),
    )
    print(
        "test: ",
        len(test_ds),
        _class_counts(split["test_labels"]),
    )

    print("first training batch:")
    print("  fc shape: ", tuple(fc.shape))
    print("  y shape:  ", tuple(y.shape))
    print("  sid shape:", tuple(sid.shape))
    print("  fc dtype: ", fc.dtype)
    print("  fc min:   ", float(fc.min()))
    print("  fc max:   ", float(fc.max()))
    print("  subject IDs:", sid.tolist())
    print("  labels:     ", y.tolist())

    print(
        "\nNo temporal HDF5 was used.\n"
        "No random split was generated.\n"
        "No FC transformation was applied.\n"
        "Subjects were resolved against the static HDF5 by subject ID."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Static ABIDE Schaefer100 dataset loader for BQN."
    )

    parser.add_argument(
        "--h5",
        default="derivatives/abide_data_tr.h5",
        help="Old static ABIDE HDF5.",
    )

    parser.add_argument(
        "--split",
        required=True,
        help="One frozen split NPZ, e.g. splits/abide_split_seed42.npz",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    _self_test(
        h5_path=args.h5,
        split_path=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
