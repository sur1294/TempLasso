#!/usr/bin/env python3

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class ABIDETemporalDataset(Dataset):

    def __init__(
        self,
        temporal_h5,
        split_npz,
        split="train",
    ):
        """
        Parameters
        ----------
        temporal_h5 : str
            Path to Step-2 temporal FC HDF5.

        split_npz : str
            Frozen split file, e.g.
            splits/abide_split_seed42.npz

        split : str
            'train', 'val', or 'test'
        """

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of: train, val, test"
            )

        self.temporal_h5 = temporal_h5
        self.split = split

        # ---------------------------------------------------------
        # Read metadata from temporal HDF5
        # ---------------------------------------------------------

        with h5py.File(self.temporal_h5, "r") as f:

            self.all_subject_ids = (
                f["subject_id"][:]
                .astype(np.int64)
            )

            self.all_labels = (
                f["label"][:]
                .astype(np.int64)
            )

            self.offsets = (
                f["offsets"][:]
                .astype(np.int64)
            )

            self.K = (
                f["K"][:]
                .astype(np.int64)
            )

            self.n_edges = int(
                f.attrs["n_edges"]
            )

            self.n_rois = int(
                f.attrs["n_rois"]
            )

        # ---------------------------------------------------------
        # Basic HDF5 sanity checks
        # ---------------------------------------------------------

        n_subjects = len(self.all_subject_ids)

        if len(self.all_labels) != n_subjects:
            raise RuntimeError(
                "subject_id and label lengths differ"
            )

        if len(self.K) != n_subjects:
            raise RuntimeError(
                "subject_id and K lengths differ"
            )

        if len(self.offsets) != n_subjects + 1:
            raise RuntimeError(
                "offsets must have length N+1"
            )

        if len(np.unique(self.all_subject_ids)) != n_subjects:
            raise RuntimeError(
                "duplicate subject IDs in temporal HDF5"
            )

        if not np.isin(self.all_labels, [0, 1]).all():
            raise RuntimeError(
                f"unexpected labels found: "
                f"{np.unique(self.all_labels)}"
            )

        # ---------------------------------------------------------
        # Build subject ID -> temporal HDF5 row index
        # ---------------------------------------------------------

        self.id_to_index = {
            int(sid): i
            for i, sid in enumerate(self.all_subject_ids)
        }

        # ---------------------------------------------------------
        # Load frozen split
        # ---------------------------------------------------------

        split_data = np.load(split_npz)

        id_key = f"{split}_ids"
        label_key = f"{split}_labels"

        if id_key not in split_data:
            raise KeyError(
                f"{id_key} not found in {split_npz}"
            )

        if label_key not in split_data:
            raise KeyError(
                f"{label_key} not found in {split_npz}"
            )

        self.subject_ids = (
            split_data[id_key]
            .astype(np.int64)
        )

        self.split_labels = (
            split_data[label_key]
            .astype(np.int64)
        )

        if len(self.subject_ids) != len(self.split_labels):
            raise RuntimeError(
                "split subject IDs and labels differ in length"
            )

        # ---------------------------------------------------------
        # Verify all split subjects exist in temporal HDF5
        # ---------------------------------------------------------

        missing = [
            int(sid)
            for sid in self.subject_ids
            if int(sid) not in self.id_to_index
        ]

        if missing:
            raise RuntimeError(
                f"{len(missing)} subjects from {split} split "
                f"missing in temporal HDF5. "
                f"Examples: {missing[:20]}"
            )

        # ---------------------------------------------------------
        # Convert split IDs -> temporal HDF5 positions
        # ---------------------------------------------------------

        self.indices = np.array(
            [
                self.id_to_index[int(sid)]
                for sid in self.subject_ids
            ],
            dtype=np.int64,
        )

        # ---------------------------------------------------------
        # Verify labels match split file
        # ---------------------------------------------------------

        mismatches = []

        for sid, expected_label, idx in zip(
            self.subject_ids,
            self.split_labels,
            self.indices,
        ):

            actual_label = int(
                self.all_labels[idx]
            )

            if actual_label != int(expected_label):
                mismatches.append(
                    (
                        int(sid),
                        int(expected_label),
                        actual_label,
                    )
                )

        if mismatches:
            raise RuntimeError(
                f"{len(mismatches)} label mismatches in {split}. "
                f"Examples: {mismatches[:20]}"
            )

        # ---------------------------------------------------------
        # HDF5 handle opened lazily
        # ---------------------------------------------------------

        self._h5 = None

        split_K = self.K[self.indices]

        print(
            f"{split:5s} | "
            f"N={len(self.indices):4d} | "
            f"K min={split_K.min():2d} "
            f"max={split_K.max():2d} "
            f"mean={split_K.mean():.2f}"
        )

    def _get_h5(self):

        if self._h5 is None:
            self._h5 = h5py.File(
                self.temporal_h5,
                "r",
            )

        return self._h5

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, item):

        f = self._get_h5()

        temporal_idx = int(
            self.indices[item]
        )

        sid = int(
            self.all_subject_ids[temporal_idx]
        )

        label = int(
            self.all_labels[temporal_idx]
        )

        start = int(
            self.offsets[temporal_idx]
        )

        end = int(
            self.offsets[temporal_idx + 1]
        )

        x = f["fc_z_upper"][
            start:end
        ].astype(np.float32)

        K_expected = int(
            self.K[temporal_idx]
        )

        # ---------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------

        if x.ndim != 2:
            raise RuntimeError(
                f"subject {sid}: expected 2D data, got {x.shape}"
            )

        if x.shape[0] != K_expected:
            raise RuntimeError(
                f"subject {sid}: "
                f"K stored={K_expected}, "
                f"loaded={x.shape[0]}"
            )

        if x.shape[1] != self.n_edges:
            raise RuntimeError(
                f"subject {sid}: "
                f"expected {self.n_edges} edges, "
                f"got {x.shape[1]}"
            )

        if not np.isfinite(x).all():
            raise RuntimeError(
                f"subject {sid}: NaN/Inf found"
            )

        return {
            "x": torch.from_numpy(x),
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "subject_id": sid,
            "K": K_expected,
        }

    def __getstate__(self):
        """
        Needed when DataLoader uses multiprocessing workers.
        Do not pickle an open HDF5 handle.
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):

        h5 = getattr(self, "_h5", None)

        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass


def temporal_collate_fn(batch):
    """
    Pad variable-length subjects only inside this mini-batch.

    Example:
        K = [12, 17, 26, 15]

    output x:
        (4, 26, 4950)

    valid_mask:
        True  = real temporal window
        False = padded position

    padding_mask:
        True  = padding
        False = real temporal window

    padding_mask is suitable for a Transformer.
    """

    batch_size = len(batch)

    lengths = torch.tensor(
        [item["K"] for item in batch],
        dtype=torch.long,
    )

    K_max = int(
        lengths.max().item()
    )

    n_edges = batch[0]["x"].shape[1]

    x_padded = torch.zeros(
        batch_size,
        K_max,
        n_edges,
        dtype=torch.float32,
    )

    valid_mask = torch.zeros(
        batch_size,
        K_max,
        dtype=torch.bool,
    )

    labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    subject_ids = []

    for i, item in enumerate(batch):

        x = item["x"]

        K = int(
            item["K"]
        )

        x_padded[i, :K] = x

        valid_mask[i, :K] = True

        labels[i] = item["label"]

        subject_ids.append(
            item["subject_id"]
        )

    padding_mask = ~valid_mask

    return {
        "x": x_padded,
        "label": labels,
        "subject_id": subject_ids,
        "lengths": lengths,
        "valid_mask": valid_mask,
        "padding_mask": padding_mask,
    }


def verify_split_files(split_npz):
    """
    Verify train/val/test subject IDs do not overlap.
    """

    d = np.load(split_npz)

    train_ids = set(
        d["train_ids"].astype(int).tolist()
    )

    val_ids = set(
        d["val_ids"].astype(int).tolist()
    )

    test_ids = set(
        d["test_ids"].astype(int).tolist()
    )

    if train_ids & val_ids:
        raise RuntimeError(
            "train/val overlap"
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "train/test overlap"
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "val/test overlap"
        )

    print(
        "Split overlap check: OK"
    )


def make_loaders(
    temporal_h5,
    split_npz,
    batch_size=16,
    num_workers=0,
):
    """
    Create train / validation / test DataLoaders.
    """

    verify_split_files(
        split_npz
    )

    train_ds = ABIDETemporalDataset(
        temporal_h5,
        split_npz,
        split="train",
    )

    val_ds = ABIDETemporalDataset(
        temporal_h5,
        split_npz,
        split="val",
    )

    test_ds = ABIDETemporalDataset(
        temporal_h5,
        split_npz,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


if __name__ == "__main__":

    TEMPORAL_H5 = (
        "derivatives/"
        "abide_schaefer100_tr2_w44_s22_temporalfc.h5"
    )

    SPLIT_NPZ = (
        "splits/"
        "abide_split_seed42.npz"
    )

    train_loader, val_loader, test_loader = make_loaders(
        temporal_h5=TEMPORAL_H5,
        split_npz=SPLIT_NPZ,
        batch_size=8,
        num_workers=0,
    )

    print("\nTesting one training batch...")

    batch = next(
        iter(train_loader)
    )

    print(
        "x shape:",
        batch["x"].shape,
    )

    print(
        "labels shape:",
        batch["label"].shape,
    )

    print(
        "lengths:",
        batch["lengths"].tolist(),
    )

    print(
        "valid mask shape:",
        batch["valid_mask"].shape,
    )

    print(
        "padding mask shape:",
        batch["padding_mask"].shape,
    )

    print(
        "subject IDs:",
        batch["subject_id"],
    )

    # ---------------------------------------------------------
    # Verify padding mask for every subject
    # ---------------------------------------------------------

    for i, K in enumerate(
        batch["lengths"].tolist()
    ):

        if not batch["valid_mask"][i, :K].all():
            raise RuntimeError(
                f"subject {i}: real windows incorrectly masked"
            )

        if batch["valid_mask"][i, K:].any():
            raise RuntimeError(
                f"subject {i}: padded windows marked as real"
            )

        if batch["padding_mask"][i, :K].any():
            raise RuntimeError(
                f"subject {i}: real windows marked as padding"
            )

        if not batch["padding_mask"][i, K:].all():
            raise RuntimeError(
                f"subject {i}: padding mask incorrect"
            )

    print("\nBatch padding/mask checks: OK")