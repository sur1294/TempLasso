from __future__ import annotations
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from unified_bn.data.common import (
    TemporalSample, canonicalize_subject_ids, fisher_z_upper_to_correlation,
    index_subject_ids, infer_n_rois_from_upper_edges, normalize_labels,
)
from unified_bn.data.schema import (
    LABEL_ALIASES, LENGTH_ALIASES, OFFSET_ALIASES, SUBJECT_ID_ALIASES,
    TEMPORAL_EDGE_ALIASES, TRIU_COL_ALIASES, TRIU_ROW_ALIASES,
    TemporalH5Schema, resolve_dataset_path, resolve_optional_dataset_path,
)

TemporalLayout = Literal["flat_lengths", "flat_offsets", "padded"]


class TemporalABIDEDataset(Dataset[TemporalSample]):
    """
    Lazy reader for temporal ABIDE Fisher-z upper-triangle FC edges.

    Supports the actual canonical artifact:
      subject_id [N]
      label      [N]
      K          [N]
      offsets    [N+1]
      fc_z_upper [W,E]
      triu_row   [E]
      triu_col   [E]

    as well as the previously tested generic flat/padded layouts.

    Edge order is scientifically validated. If explicit triu_row/triu_col are
    present, they must exactly equal the canonical np.triu_indices(R, k=1)
    ordering used by fisher_z_upper_to_correlation. We fail rather than silently
    reconstruct edges in the wrong anatomical positions.
    """

    def __init__(
        self, path: str | Path, *, schema: TemporalH5Schema | None = None,
        subject_ids: Sequence[str] | None = None,
        expected_n_rois: int | None = 100,
        dtype: torch.dtype = torch.float32,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.path = Path(path)
        self.schema = schema or TemporalH5Schema()
        self.expected_n_rois = expected_n_rois
        self.dtype = dtype
        self._handle: h5py.File | None = None

        if not self.path.is_file():
            raise FileNotFoundError(f"Temporal HDF5 not found: {self.path}")

        with h5py.File(self.path, "r") as h5:
            self.subject_id_key = resolve_dataset_path(
                h5, role="subject IDs", override=self.schema.subject_ids,
                aliases=SUBJECT_ID_ALIASES,
            )
            self.label_key = resolve_dataset_path(
                h5, role="labels", override=self.schema.labels,
                aliases=LABEL_ALIASES,
            )
            self.edge_key = resolve_dataset_path(
                h5, role="temporal Fisher-z edges", override=self.schema.edges,
                aliases=TEMPORAL_EDGE_ALIASES,
            )
            self.length_key = resolve_optional_dataset_path(
                h5, role="temporal lengths", override=self.schema.lengths,
                aliases=LENGTH_ALIASES,
            )
            self.offset_key = resolve_optional_dataset_path(
                h5, role="temporal offsets", override=self.schema.offsets,
                aliases=OFFSET_ALIASES,
            )
            self.triu_row_key = resolve_optional_dataset_path(
                h5, role="upper-triangle row indices",
                override=self.schema.triu_row, aliases=TRIU_ROW_ALIASES,
            )
            self.triu_col_key = resolve_optional_dataset_path(
                h5, role="upper-triangle column indices",
                override=self.schema.triu_col, aliases=TRIU_COL_ALIASES,
            )

            if (self.triu_row_key is None) != (self.triu_col_key is None):
                raise ValueError(
                    "Temporal HDF5 must contain both triu_row and triu_col "
                    "or neither; found only one."
                )

            all_ids = canonicalize_subject_ids(h5[self.subject_id_key][...])
            all_labels = normalize_labels(num_classes=self.num_classes, values=h5[self.label_key][...])
            if len(all_labels) != len(all_ids):
                raise ValueError(
                    f"Temporal subject-axis mismatch: ids={len(all_ids)}, "
                    f"labels={len(all_labels)}."
                )

            edge_ds = h5[self.edge_key]
            if edge_ds.ndim == 2:
                n_edges = int(edge_ds.shape[1])
                lengths, offsets, layout = self._resolve_flat_layout(
                    h5, len(all_ids), int(edge_ds.shape[0])
                )
            elif edge_ds.ndim == 3:
                n_edges = int(edge_ds.shape[2])
                lengths, offsets, layout = self._resolve_padded_layout(
                    h5, len(all_ids), int(edge_ds.shape[0]), int(edge_ds.shape[1])
                )
            else:
                raise ValueError(
                    f"Temporal edges must be [W,E] or [N,K,E], got {edge_ds.shape}."
                )

            n_rois = infer_n_rois_from_upper_edges(n_edges)
            if expected_n_rois is not None and n_rois != int(expected_n_rois):
                raise ValueError(
                    f"Expected R={expected_n_rois}, but E={n_edges} "
                    f"implies R={n_rois}."
                )

            self._validate_upper_triangle_order(
                h5, n_rois=n_rois, n_edges=n_edges
            )

        self.n_rois, self.n_edges, self.layout = n_rois, n_edges, layout
        self._all_subject_ids = all_ids
        self._all_labels = all_labels
        self._all_lengths = lengths
        self._offsets = offsets

        if subject_ids is None:
            self._indices = np.arange(len(all_ids), dtype=np.int64)
        else:
            self._indices = index_subject_ids(
                all_ids, tuple(str(x).strip() for x in subject_ids)
            )

        self.subject_ids = tuple(all_ids[i] for i in self._indices)
        self.labels = np.asarray(
            [all_labels[i] for i in self._indices], dtype=np.int64
        )
        self.lengths = np.asarray(
            [lengths[i] for i in self._indices], dtype=np.int64
        )

    @staticmethod
    def _read_int_vector(h5: h5py.File, key: str, role: str) -> np.ndarray:
        arr = np.asarray(h5[key][...]).reshape(-1)
        if not np.issubdtype(arr.dtype, np.integer):
            if (
                np.issubdtype(arr.dtype, np.floating)
                and np.isfinite(arr).all()
                and np.array_equal(arr, np.rint(arr))
            ):
                arr = np.rint(arr)
            else:
                raise TypeError(f"{role} must be integer-valued.")
        return arr.astype(np.int64, copy=False)

    def _validate_upper_triangle_order(
        self, h5: h5py.File, *, n_rois: int, n_edges: int
    ) -> None:
        if self.triu_row_key is None:
            return

        rows = self._read_int_vector(
            h5, self.triu_row_key, "Upper-triangle row indices"
        )
        cols = self._read_int_vector(
            h5, self.triu_col_key, "Upper-triangle column indices"
        )

        if rows.size != n_edges or cols.size != n_edges:
            raise ValueError(
                "triu_row/triu_col length must equal temporal edge dimension E: "
                f"rows={rows.size}, cols={cols.size}, E={n_edges}."
            )
        if np.any(rows < 0) or np.any(cols < 0):
            raise ValueError("Upper-triangle indices cannot be negative.")
        if np.any(rows >= n_rois) or np.any(cols >= n_rois):
            raise ValueError("Upper-triangle indices exceed ROI dimension.")
        if np.any(rows >= cols):
            raise ValueError(
                "Expected strict upper-triangle indices with row < column."
            )

        expected_rows, expected_cols = np.triu_indices(n_rois, k=1)
        if not (
            np.array_equal(rows, expected_rows)
            and np.array_equal(cols, expected_cols)
        ):
            first_bad = np.flatnonzero(
                (rows != expected_rows) | (cols != expected_cols)
            )
            j = int(first_bad[0]) if first_bad.size else -1
            raise ValueError(
                "Stored triu_row/triu_col ordering does not match the canonical "
                "upper-triangle ordering used by reconstruction. "
                f"First mismatch at edge {j}: stored=({rows[j]},{cols[j]}), "
                f"expected=({expected_rows[j]},{expected_cols[j]})."
            )

    def _resolve_flat_layout(self, h5, n_subjects, n_windows):
        lengths = offsets = None

        if self.length_key is not None:
            lengths = self._read_int_vector(
                h5, self.length_key, "Temporal lengths"
            )
            if lengths.size != n_subjects:
                raise ValueError(
                    f"Expected {n_subjects} lengths, found {lengths.size}."
                )
            if np.any(lengths <= 0):
                raise ValueError("All temporal lengths must be positive.")
            if int(lengths.sum()) != n_windows:
                raise ValueError(
                    f"sum(lengths)={int(lengths.sum())} != W={n_windows}."
                )

        if self.offset_key is not None:
            offsets = self._read_int_vector(
                h5, self.offset_key, "Temporal offsets"
            )
            if offsets.size != n_subjects + 1:
                raise ValueError(
                    f"Expected N+1={n_subjects+1} offsets, found {offsets.size}."
                )
            if offsets[0] != 0 or offsets[-1] != n_windows:
                raise ValueError(
                    "Temporal offsets must start at 0 and end at W."
                )
            if np.any(np.diff(offsets) <= 0):
                raise ValueError(
                    "Temporal offsets must be strictly increasing."
                )

        if lengths is None and offsets is None:
            raise ValueError(
                "Flat temporal edges [W,E] require lengths [N] or offsets [N+1]."
            )

        if offsets is None:
            assert lengths is not None
            offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
            layout = "flat_lengths"
        elif lengths is None:
            lengths = np.diff(offsets)
            layout = "flat_offsets"
        else:
            implied = np.diff(offsets)
            if not np.array_equal(implied, lengths):
                mismatch = np.flatnonzero(implied != lengths)
                j = int(mismatch[0])
                raise ValueError(
                    "Temporal K/lengths and offsets disagree. "
                    f"First mismatch subject index {j}: "
                    f"K={int(lengths[j])}, offset_diff={int(implied[j])}."
                )
            layout = "flat_lengths"

        return lengths, offsets, layout

    def _resolve_padded_layout(
        self, h5, n_subjects, padded_subjects, padded_steps
    ):
        if padded_subjects != n_subjects:
            raise ValueError(
                f"Padded temporal subject axis={padded_subjects}, ids N={n_subjects}."
            )
        if self.length_key is None:
            raise ValueError("Padded temporal edges require lengths [N].")

        lengths = self._read_int_vector(
            h5, self.length_key, "Temporal lengths"
        )
        if lengths.size != n_subjects:
            raise ValueError(
                f"Expected {n_subjects} lengths, found {lengths.size}."
            )
        if np.any(lengths <= 0):
            raise ValueError("All temporal lengths must be positive.")
        if np.any(lengths > padded_steps):
            raise ValueError("At least one temporal length exceeds padded K.")

        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        return lengths, offsets, "padded"

    def __len__(self) -> int:
        return int(self._indices.size)

    @property
    def min_windows(self) -> int:
        return int(self.lengths.min())

    @property
    def max_windows(self) -> int:
        return int(self.lengths.max())

    @property
    def mean_windows(self) -> float:
        return float(self.lengths.mean())

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> TemporalSample:
        source_index = int(self._indices[index])
        length = int(self._all_lengths[source_index])
        edge_ds = self._get_handle()[self.edge_key]

        if self.layout == "padded":
            edges = np.asarray(edge_ds[source_index, :length, :])
        else:
            start = int(self._offsets[source_index])
            stop = int(self._offsets[source_index + 1])
            if stop - start != length:
                raise RuntimeError(
                    "Internal temporal offset/length inconsistency."
                )
            edges = np.asarray(edge_ds[start:stop, :])

        expected = (length, self.n_edges)
        if edges.shape != expected:
            raise RuntimeError(
                f"Malformed temporal slice {edges.shape}; expected {expected}."
            )
        if not np.issubdtype(edges.dtype, np.floating):
            raise TypeError(
                f"Temporal Fisher-z edges must be floating point, "
                f"found {edges.dtype}."
            )
        if not np.isfinite(edges).all():
            raise ValueError(
                f"Temporal Fisher-z contains NaN/Inf for "
                f"{self._all_subject_ids[source_index]}."
            )

        dynamic_fc = fisher_z_upper_to_correlation(
            edges, n_rois=self.n_rois, dtype=self.dtype
        )

        return TemporalSample(
            self._all_subject_ids[source_index],
            int(self._all_labels[source_index]),
            dynamic_fc,
            length,
            source_index,
        )

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()
