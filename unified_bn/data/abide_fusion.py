from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from unified_bn.data.abide_static import StaticABIDEDataset
from unified_bn.data.abide_temporal import TemporalABIDEDataset
from unified_bn.data.alignment import require_aligned_modalities
from unified_bn.data.common import canonicalize_subject_ids, index_subject_ids
from unified_bn.data.schema import StaticH5Schema, TemporalH5Schema


@dataclass(frozen=True)
class FusionSample:
    """One subject with aligned static and temporal FC inputs."""

    subject_id: str
    label: int
    static_fc: torch.Tensor
    dynamic_fc: torch.Tensor
    length: int
    static_source_index: int
    temporal_source_index: int


class FusionABIDEDataset(Dataset[FusionSample]):
    """
    ID-aligned joint static/temporal ABIDE dataset.

    Alignment is always performed by canonical subject ID. Static and temporal
    storage positions are never assumed to correspond. The two complete source
    artifacts are first checked for identical subject sets and labels, then an
    optional requested subject order is mapped independently into each source.

    This keeps frozen split order authoritative while preserving each source
    artifact's original row/group index for provenance.
    """

    def __init__(
        self,
        static_path: Union[str, Path],
        temporal_path: Union[str, Path],
        *,
        static_schema: Optional[StaticH5Schema] = None,
        temporal_schema: Optional[TemporalH5Schema] = None,
        subject_ids: Optional[Sequence[object]] = None,
        expected_n_rois: Optional[int] = 100,
        dtype: torch.dtype = torch.float32,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

        self.static_path = Path(static_path)
        self.temporal_path = Path(temporal_path)
        self.expected_n_rois = expected_n_rois
        self.dtype = dtype

        self.static_dataset = StaticABIDEDataset(
            self.static_path,
            schema=static_schema,
            subject_ids=None,
            expected_n_rois=expected_n_rois,
            dtype=dtype,
            num_classes=num_classes,
        )
        self.temporal_dataset = TemporalABIDEDataset(
            self.temporal_path,
            schema=temporal_schema,
            subject_ids=None,
            expected_n_rois=expected_n_rois,
            dtype=dtype,
            num_classes=num_classes,
        )

        if self.static_dataset.n_rois != self.temporal_dataset.n_rois:
            raise ValueError(
                "Static/temporal ROI counts differ: "
                f"{self.static_dataset.n_rois} vs "
                f"{self.temporal_dataset.n_rois}."
            )

        # Fail closed on any source-artifact subject or label mismatch before
        # building a requested split subset.
        require_aligned_modalities(
            self.static_dataset.subject_ids,
            self.static_dataset.labels,
            self.temporal_dataset.subject_ids,
            self.temporal_dataset.labels,
        )

        if subject_ids is None:
            requested = tuple(self.static_dataset.subject_ids)
        else:
            requested = canonicalize_subject_ids(subject_ids)

        self._static_indices = index_subject_ids(
            self.static_dataset.subject_ids,
            requested,
        )
        self._temporal_indices = index_subject_ids(
            self.temporal_dataset.subject_ids,
            requested,
        )

        self.subject_ids = requested
        self.n_rois = int(self.static_dataset.n_rois)

        static_labels = np.asarray(
            [self.static_dataset.labels[i] for i in self._static_indices],
            dtype=np.int64,
        )
        temporal_labels = np.asarray(
            [self.temporal_dataset.labels[i] for i in self._temporal_indices],
            dtype=np.int64,
        )
        if not np.array_equal(static_labels, temporal_labels):
            mismatch = np.flatnonzero(static_labels != temporal_labels)
            first = int(mismatch[0])
            raise ValueError(
                "Static/temporal labels disagree after ID alignment for "
                f"subject {requested[first]!r}."
            )

        self.labels = static_labels
        self.lengths = np.asarray(
            [self.temporal_dataset.lengths[i] for i in self._temporal_indices],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.subject_ids)

    @property
    def min_windows(self) -> int:
        return int(self.lengths.min())

    @property
    def max_windows(self) -> int:
        return int(self.lengths.max())

    @property
    def mean_windows(self) -> float:
        return float(self.lengths.mean())

    def __getitem__(self, index: int) -> FusionSample:
        static_index = int(self._static_indices[index])
        temporal_index = int(self._temporal_indices[index])

        static_sample = self.static_dataset[static_index]
        temporal_sample = self.temporal_dataset[temporal_index]
        expected_id = self.subject_ids[index]

        if static_sample.subject_id != expected_id:
            raise RuntimeError(
                "Static fusion lookup returned the wrong subject: "
                f"expected={expected_id!r}, got={static_sample.subject_id!r}."
            )
        if temporal_sample.subject_id != expected_id:
            raise RuntimeError(
                "Temporal fusion lookup returned the wrong subject: "
                f"expected={expected_id!r}, got={temporal_sample.subject_id!r}."
            )
        if static_sample.label != temporal_sample.label:
            raise RuntimeError(
                "Static/temporal labels disagree at read time for "
                f"subject {expected_id!r}: "
                f"static={static_sample.label}, temporal={temporal_sample.label}."
            )
        if static_sample.label != int(self.labels[index]):
            raise RuntimeError(
                "Fusion dataset label changed after construction for "
                f"subject {expected_id!r}."
            )
        if tuple(static_sample.fc.shape) != (self.n_rois, self.n_rois):
            raise RuntimeError(
                "Static FC shape changed after fusion dataset construction."
            )
        if temporal_sample.dynamic_fc.ndim != 3 or tuple(
            temporal_sample.dynamic_fc.shape[1:]
        ) != (self.n_rois, self.n_rois):
            raise RuntimeError(
                "Temporal FC shape changed after fusion dataset construction."
            )

        return FusionSample(
            subject_id=expected_id,
            label=static_sample.label,
            static_fc=static_sample.fc,
            dynamic_fc=temporal_sample.dynamic_fc,
            length=temporal_sample.length,
            static_source_index=static_sample.source_index,
            temporal_source_index=temporal_sample.source_index,
        )

    def close(self) -> None:
        self.static_dataset.close()
        self.temporal_dataset.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
