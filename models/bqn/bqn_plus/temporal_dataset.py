#!/usr/bin/env python3
"""
Temporal ABIDE Schaefer100 dataset utilities for BQN+.

Uses ONLY:
  derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5
and one frozen subject-ID split NPZ.

The temporal HDF5 produced by build_step2_temporal_fc.py stores:
  fc_z_upper  : [sum_i K_i, 4950] Fisher-z upper triangle
  offsets     : [N+1]
  K           : [N]
  subject_id  : [N]
  label       : [N]
  triu_row    : [4950]
  triu_col    : [4950]

For BQN+, this loader reconstructs each full symmetric Fisher-z matrix and
inverts Fisher-z with tanh so that the shared BQN backbone receives
correlation-scale matrices in the dynamic branch:

  [K,4950] z-upper -> [K,100,100] z -> tanh -> correlation, diag=1

Native K is preserved. Padding occurs only inside each batch.
No ComBat, scaling, standardization, truncation, or new split generation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EXPECTED_N_SUBJECTS = 1025
EXPECTED_N_ROIS = 100
EXPECTED_N_EDGES = 4950
EXPECTED_SPLIT_SIZES = {"train": 717, "val": 103, "test": 205}


@dataclass(frozen=True)
class TemporalSubjectRecord:
    subject_id: int
    label: int
    start: int
    stop: int
    k: int


def load_frozen_split(split_path: str) -> Dict[str, np.ndarray]:
    data = np.load(split_path, allow_pickle=False)
    required = {
        "train_ids", "val_ids", "test_ids",
        "train_labels", "val_labels", "test_labels",
    }
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"{split_path} missing keys: {sorted(missing)}")

    out = {
        "train_ids": np.asarray(data["train_ids"], dtype=np.int64),
        "val_ids": np.asarray(data["val_ids"], dtype=np.int64),
        "test_ids": np.asarray(data["test_ids"], dtype=np.int64),
        "train_labels": np.asarray(data["train_labels"], dtype=np.int64),
        "val_labels": np.asarray(data["val_labels"], dtype=np.int64),
        "test_labels": np.asarray(data["test_labels"], dtype=np.int64),
    }
    if "seed" in data.files:
        out["seed"] = np.asarray(data["seed"], dtype=np.int64)
    return out


def reconstruct_correlation(z_upper, triu_row, triu_col, n_rois):
    """Reconstruct full correlation matrices from Fisher-z upper triangles."""
    z_upper = np.asarray(z_upper, dtype=np.float32)
    triu_row = np.asarray(triu_row, dtype=np.int64)
    triu_col = np.asarray(triu_col, dtype=np.int64)

    if z_upper.ndim != 2:
        raise ValueError(f"Expected [K,E], got {z_upper.shape}")
    k, e = z_upper.shape
    if len(triu_row) != e or len(triu_col) != e:
        raise ValueError("Triangle index length does not match edge count")

    z = np.zeros((k, n_rois, n_rois), dtype=np.float32)
    z[:, triu_row, triu_col] = z_upper
    z[:, triu_col, triu_row] = z_upper

    corr = np.tanh(z).astype(np.float32, copy=False)
    d = np.arange(n_rois)
    corr[:, d, d] = 1.0
    return corr


class TemporalABIDESchaeferDataset(Dataset):
    """
    Dataset for a fixed subject-ID list.

    Each item returns:
      dynamic_fc : float32 [K_i,100,100]
      label      : long scalar
      subject_id : long scalar
      length     : long scalar
    """

    def __init__(self, temporal_h5: str, subject_ids, expected_labels):
        self.temporal_h5 = str(temporal_h5)
        self.subject_ids = np.asarray(subject_ids, dtype=np.int64)
        self.expected_labels = np.asarray(expected_labels, dtype=np.int64)

        if self.subject_ids.ndim != 1 or self.expected_labels.ndim != 1:
            raise ValueError("subject_ids and expected_labels must be 1-D")
        if len(self.subject_ids) != len(self.expected_labels):
            raise ValueError("subject_ids / expected_labels length mismatch")
        if len(np.unique(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("Duplicate requested subject IDs")

        self.records, self.n_rois, self.n_edges, self.triu_row, self.triu_col = self._build_records()

    def _build_records(self):
        with h5py.File(self.temporal_h5, "r") as f:
            required = {
                "fc_z_upper", "offsets", "K", "subject_id", "label",
                "triu_row", "triu_col",
            }
            missing = required - set(f.keys())
            if missing:
                raise RuntimeError(f"Temporal HDF5 missing datasets: {sorted(missing)}")

            file_ids = np.asarray(f["subject_id"][:], dtype=np.int64)
            file_labels = np.asarray(f["label"][:], dtype=np.int64)
            offsets = np.asarray(f["offsets"][:], dtype=np.int64)
            ks = np.asarray(f["K"][:], dtype=np.int64)
            triu_row = np.asarray(f["triu_row"][:], dtype=np.int64)
            triu_col = np.asarray(f["triu_col"][:], dtype=np.int64)

            n_rois = int(f.attrs["n_rois"])
            n_edges = int(f.attrs["n_edges"])

            if n_rois != EXPECTED_N_ROIS:
                raise RuntimeError(f"Expected {EXPECTED_N_ROIS} ROIs, found {n_rois}")
            if n_edges != EXPECTED_N_EDGES:
                raise RuntimeError(f"Expected {EXPECTED_N_EDGES} edges, found {n_edges}")
            if len(file_ids) != EXPECTED_N_SUBJECTS:
                raise RuntimeError(f"Expected {EXPECTED_N_SUBJECTS} subjects, found {len(file_ids)}")
            if len(np.unique(file_ids)) != len(file_ids):
                raise RuntimeError("Duplicate subject IDs in temporal HDF5")
            if offsets.shape != (len(file_ids) + 1,):
                raise RuntimeError("offsets has invalid shape")
            if ks.shape != (len(file_ids),):
                raise RuntimeError("K has invalid shape")
            if not np.array_equal(np.diff(offsets), ks):
                raise RuntimeError("K does not match differences of offsets")
            if int(offsets[-1]) != int(f["fc_z_upper"].shape[0]):
                raise RuntimeError("Final offset != total temporal windows")
            if int(f["fc_z_upper"].shape[1]) != n_edges:
                raise RuntimeError("fc_z_upper edge count mismatch")
            if len(triu_row) != n_edges or len(triu_col) != n_edges:
                raise RuntimeError("Triangle index length mismatch")

            id_to_index = {int(sid): i for i, sid in enumerate(file_ids)}
            records: List[TemporalSubjectRecord] = []

            for sid, expected_label in zip(self.subject_ids.tolist(), self.expected_labels.tolist()):
                if sid not in id_to_index:
                    raise KeyError(f"Split subject {sid} missing from temporal HDF5")
                i = id_to_index[sid]
                file_label = int(file_labels[i])
                if file_label != int(expected_label):
                    raise ValueError(
                        f"Label mismatch for subject {sid}: split={expected_label}, h5={file_label}"
                    )

                start = int(offsets[i])
                stop = int(offsets[i + 1])
                k = int(ks[i])
                if stop - start != k:
                    raise RuntimeError(f"Subject {sid}: offset span != K")
                if k <= 0:
                    raise RuntimeError(f"Subject {sid}: invalid K={k}")

                records.append(TemporalSubjectRecord(int(sid), file_label, start, stop, k))

        return records, n_rois, n_edges, triu_row, triu_col

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        with h5py.File(self.temporal_h5, "r") as f:
            z_upper = np.asarray(
                f["fc_z_upper"][rec.start:rec.stop], dtype=np.float32
            )

        if z_upper.shape != (rec.k, self.n_edges):
            raise RuntimeError(
                f"Subject {rec.subject_id}: expected {(rec.k, self.n_edges)}, got {z_upper.shape}"
            )
        if not np.isfinite(z_upper).all():
            raise RuntimeError(f"Subject {rec.subject_id}: non-finite Fisher-z values")

        corr = reconstruct_correlation(
            z_upper, self.triu_row, self.triu_col, self.n_rois
        )

        if not np.isfinite(corr).all():
            raise RuntimeError(f"Subject {rec.subject_id}: non-finite reconstructed FC")
        if np.max(np.abs(corr - corr.transpose(0, 2, 1))) > 1e-6:
            raise RuntimeError(f"Subject {rec.subject_id}: reconstructed FC not symmetric")
        if not np.allclose(np.diagonal(corr, axis1=1, axis2=2), 1.0, atol=1e-6):
            raise RuntimeError(f"Subject {rec.subject_id}: diagonal != 1")
        if corr.min() < -1.000001 or corr.max() > 1.000001:
            raise RuntimeError(f"Subject {rec.subject_id}: FC outside [-1,1]")

        return (
            torch.from_numpy(corr),
            torch.tensor(rec.label, dtype=torch.long),
            torch.tensor(rec.subject_id, dtype=torch.long),
            torch.tensor(rec.k, dtype=torch.long),
        )


def temporal_collate_fn(batch):
    """Batch-only padding. valid_mask=True denotes a real temporal window."""
    if not batch:
        raise ValueError("Empty batch")

    fcs, labels, subject_ids, lengths = zip(*batch)
    labels = torch.stack(labels)
    subject_ids = torch.stack(subject_ids)
    lengths = torch.stack(lengths)

    b = len(fcs)
    kmax = int(lengths.max().item())
    r = int(fcs[0].shape[-1])

    dynamic_fc = torch.zeros(b, kmax, r, r, dtype=torch.float32)
    valid_mask = torch.zeros(b, kmax, dtype=torch.bool)

    for i, fc in enumerate(fcs):
        k = fc.shape[0]
        if int(lengths[i]) != k:
            raise RuntimeError("Length metadata mismatch")
        dynamic_fc[i, :k] = fc
        valid_mask[i, :k] = True

    return {
        "dynamic_fc": dynamic_fc,
        "labels": labels,
        "subject_ids": subject_ids,
        "lengths": lengths,
        "valid_mask": valid_mask,
    }


def build_temporal_datasets(temporal_h5: str, split_path: str):
    split = load_frozen_split(split_path)
    train_ds = TemporalABIDESchaeferDataset(
        temporal_h5, split["train_ids"], split["train_labels"]
    )
    val_ds = TemporalABIDESchaeferDataset(
        temporal_h5, split["val_ids"], split["val_labels"]
    )
    test_ds = TemporalABIDESchaeferDataset(
        temporal_h5, split["test_ids"], split["test_labels"]
    )
    return train_ds, val_ds, test_ds


def build_temporal_loaders(
    temporal_h5: str,
    split_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_seed: int = 42,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    train_ds, val_ds, test_ds = build_temporal_datasets(temporal_h5, split_path)
    g = torch.Generator().manual_seed(int(shuffle_seed))

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=temporal_collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def _self_test(temporal_h5, split_path, batch_size, num_workers):
    split = load_frozen_split(split_path)
    train_ds, val_ds, test_ds = build_temporal_datasets(temporal_h5, split_path)

    assert len(train_ds) == EXPECTED_SPLIT_SIZES["train"]
    assert len(val_ds) == EXPECTED_SPLIT_SIZES["val"]
    assert len(test_ds) == EXPECTED_SPLIT_SIZES["test"]

    fc0, y0, sid0, k0 = train_ds[0]
    assert fc0.shape == (int(k0), 100, 100)
    assert torch.isfinite(fc0).all()
    assert torch.allclose(fc0, fc0.transpose(-1, -2), atol=1e-6)

    train_loader, _, _ = build_temporal_loaders(
        temporal_h5, split_path, batch_size, num_workers,
        int(np.asarray(split.get("seed", np.array(42))).item()),
    )
    batch = next(iter(train_loader))

    x = batch["dynamic_fc"]
    lengths = batch["lengths"]
    mask = batch["valid_mask"]

    assert torch.equal(mask.sum(dim=1).long(), lengths)
    if (~mask).any():
        assert torch.all(x[~mask] == 0)

    valid_fc = x[mask]
    valid_diag = torch.diagonal(valid_fc, dim1=-2, dim2=-1)
    assert torch.allclose(valid_diag, torch.ones_like(valid_diag), atol=1e-6)

    print("TemporalABIDESchaeferDataset self-test: PASS")
    print(f"temporal H5: {temporal_h5}")
    print(f"split:       {split_path}")
    if "seed" in split:
        print(f"stored seed: {int(split['seed'].item())}")
    print(f"train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"first subject: sid={int(sid0)} label={int(y0)} K={int(k0)} shape={tuple(fc0.shape)}")
    print(f"first batch dynamic_fc: {tuple(x.shape)}")
    print(f"lengths: {lengths.tolist()}")
    print(f"valid counts: {mask.sum(dim=1).tolist()}")
    print(f"Kmax: {x.shape[1]}")
    print("Representation: Fisher-z upper triangle -> full symmetric z -> tanh -> correlation, diagonal=1")
    print("No truncation. No new split. No ComBat/scaling/standardization.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--temporal-h5",
        default="derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5",
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    _self_test(args.temporal_h5, args.split, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
