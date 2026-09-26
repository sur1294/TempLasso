from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from unified_bn.data.common import (
    StaticSample,
    canonical_subject_id,
    canonicalize_subject_ids,
    index_subject_ids,
    normalize_labels,
)
from unified_bn.data.schema import (
    LABEL_ALIASES,
    STATIC_FC_ALIASES,
    SUBJECT_ID_ALIASES,
    StaticH5Schema,
    resolve_dataset_path,
)

StaticLayout = Literal["dense", "subject_groups"]

_ABIDE_GROUP_RE = re.compile(r"^sub-(control|patient)(\d+)$")


def _parse_abide_subject_group(group_name: str) -> tuple[str, int]:
    """
    Parse the legacy static HDF5 subject-group name into the canonical ABIDE ID
    and binary diagnosis label.

    Examples
    --------
    sub-control50030 -> ("50030", 0)
    sub-patient51585 -> ("51585", 1)

    This normalization is evidence-based for the current ABIDE artifacts:
    removing the static-file class prefix yields a one-to-one match with all
    1025 temporal IDs and produces zero label mismatches.
    """
    match = _ABIDE_GROUP_RE.fullmatch(group_name)
    if match is None:
        raise ValueError(
            "Cannot parse hierarchical ABIDE subject group "
            f"{group_name!r}; expected exactly "
            "'sub-control<digits>' or 'sub-patient<digits>'."
        )

    diagnosis, numeric_id = match.groups()
    label = 0 if diagnosis == "control" else 1
    return numeric_id, label


def _discover_subject_group_layout(
    h5: h5py.File,
    *,
    atlas: str,
    corr_dataset: str,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray] | None:
    """
    Detect the legacy hierarchical static layout

        <group>/<atlas>/<corr_dataset>

    and return:
        canonical_subject_ids,
        source_group_names,
        labels

    Canonical IDs deliberately exclude the static-only class prefix so they
    align with temporal HDF5 IDs and the frozen split NPZ IDs.
    """
    source_groups: list[str] = []

    for name, obj in h5.items():
        if not isinstance(obj, h5py.Group):
            continue

        if _ABIDE_GROUP_RE.fullmatch(name) is None:
            continue

        path = f"{name}/{atlas}/{corr_dataset}"
        if path in h5 and isinstance(h5[path], h5py.Dataset):
            source_groups.append(name)

    if not source_groups:
        return None

    subject_like = sorted(
        name
        for name, obj in h5.items()
        if isinstance(obj, h5py.Group)
        and (
            name.lower().startswith("sub-control")
            or name.lower().startswith("sub-patient")
        )
    )

    if len(source_groups) != len(subject_like):
        missing = sorted(set(subject_like) - set(source_groups))
        raise ValueError(
            f"{len(missing)} subject groups are missing "
            f"{atlas}/{corr_dataset}: {missing[:10]}"
        )

    # Preserve the static artifact's deterministic lexical group ordering.
    # Canonicalization changes ID representation only; it must not silently
    # reorder subjects. Cross-modal matching is performed explicitly by ID.
    source_groups = sorted(source_groups)

    parsed = []
    for group_name in source_groups:
        canonical_id, label = _parse_abide_subject_group(group_name)
        parsed.append((canonical_id, group_name, label))

    canonical_ids = tuple(x[0] for x in parsed)
    source_groups = tuple(x[1] for x in parsed)
    labels = np.asarray([x[2] for x in parsed], dtype=np.int64)

    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError(
            "Duplicate canonical ABIDE IDs were produced from hierarchical "
            "static subject groups."
        )

    return canonical_ids, source_groups, labels


class StaticABIDEDataset(Dataset[StaticSample]):
    """
    Worker-safe static ABIDE reader supporting both:

      1. dense layout
           subject_ids [N], labels [N], FC [N,R,R]

      2. actual legacy hierarchical layout
           sub-controlXXXXX/schaefer100/corr_matrix
           sub-patientXXXXX/schaefer100/corr_matrix

    Public subject IDs are canonical ABIDE IDs (e.g. "50030") for both
    layouts. For hierarchical storage, the original HDF5 group name is retained
    internally for matrix lookup and provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema: StaticH5Schema | None = None,
        subject_ids: Sequence[str] | None = None,
        expected_n_rois: int | None = 100,
        dtype: torch.dtype = torch.float32,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

        self.path = Path(path)
        self.schema = schema or StaticH5Schema()
        self.expected_n_rois = expected_n_rois
        self.dtype = dtype
        self._handle: h5py.File | None = None
        self._source_groups: tuple[str, ...] | None = None

        if not self.path.is_file():
            raise FileNotFoundError(f"Static HDF5 not found: {self.path}")

        with h5py.File(self.path, "r") as h5:
            hierarchical = _discover_subject_group_layout(
                h5,
                atlas=self.schema.atlas,
                corr_dataset=self.schema.corr_dataset,
            )

            if hierarchical is not None and all(
                value is None
                for value in (
                    self.schema.subject_ids,
                    self.schema.labels,
                    self.schema.fc,
                )
            ):
                self.layout: StaticLayout = "subject_groups"
                all_ids, source_groups, all_labels = hierarchical
                self._source_groups = source_groups

                self.subject_id_key = None
                self.label_key = None
                self.fc_key = None

                first_path = (
                    f"{source_groups[0]}/{self.schema.atlas}/"
                    f"{self.schema.corr_dataset}"
                )
                first_ds = h5[first_path]
                if first_ds.ndim != 2 or first_ds.shape[0] != first_ds.shape[1]:
                    raise ValueError(
                        f"Hierarchical static FC must be square [R,R], "
                        f"got {first_ds.shape} at {first_path}."
                    )
                n_rois = int(first_ds.shape[0])

                for group_name in source_groups:
                    path_i = (
                        f"{group_name}/{self.schema.atlas}/"
                        f"{self.schema.corr_dataset}"
                    )
                    ds_i = h5[path_i]
                    if ds_i.shape != (n_rois, n_rois):
                        raise ValueError(
                            f"Inconsistent static FC shape for {group_name}: "
                            f"{ds_i.shape}, expected {(n_rois, n_rois)}."
                        )
            else:
                self.layout = "dense"

                self.subject_id_key = resolve_dataset_path(
                    h5,
                    role="subject IDs",
                    override=self.schema.subject_ids,
                    aliases=SUBJECT_ID_ALIASES,
                )
                self.label_key = resolve_dataset_path(
                    h5,
                    role="labels",
                    override=self.schema.labels,
                    aliases=LABEL_ALIASES,
                )
                self.fc_key = resolve_dataset_path(
                    h5,
                    role="static FC",
                    override=self.schema.fc,
                    aliases=STATIC_FC_ALIASES,
                )

                all_ids = canonicalize_subject_ids(
                    h5[self.subject_id_key][...]
                )
                all_labels = normalize_labels(num_classes=self.num_classes, values=
                    h5[self.label_key][...]
                )
                fc_ds = h5[self.fc_key]

                if fc_ds.ndim != 3:
                    raise ValueError(
                        f"Static FC must be [N,R,R], got {fc_ds.shape}."
                    )
                if fc_ds.shape[1] != fc_ds.shape[2]:
                    raise ValueError(
                        f"Static FC matrices must be square, got {fc_ds.shape}."
                    )
                if (
                    len(all_ids) != len(all_labels)
                    or fc_ds.shape[0] != len(all_ids)
                ):
                    raise ValueError(
                        "Static subject-axis mismatch: "
                        f"ids={len(all_ids)}, labels={len(all_labels)}, "
                        f"fc={fc_ds.shape[0]}."
                    )

                n_rois = int(fc_ds.shape[1])

            if expected_n_rois is not None and n_rois != int(expected_n_rois):
                raise ValueError(
                    f"Expected R={expected_n_rois}, found R={n_rois}."
                )

        self.n_rois = n_rois
        self._all_subject_ids = tuple(all_ids)
        self._all_labels = np.asarray(all_labels, dtype=np.int64)

        if subject_ids is None:
            self._indices = np.arange(len(all_ids), dtype=np.int64)
        else:
            requested = tuple(canonical_subject_id(x) for x in subject_ids)
            self._indices = index_subject_ids(all_ids, requested)

        self.subject_ids = tuple(all_ids[i] for i in self._indices)
        self.labels = np.asarray(
            [all_labels[i] for i in self._indices],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self._indices.size)

    def _get_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> StaticSample:
        source_index = int(self._indices[index])
        h5 = self._get_handle()

        if self.layout == "subject_groups":
            assert self._source_groups is not None
            group_name = self._source_groups[source_index]
            path = (
                f"{group_name}/{self.schema.atlas}/"
                f"{self.schema.corr_dataset}"
            )
            fc_np = np.asarray(h5[path])
        else:
            assert self.fc_key is not None
            fc_np = np.asarray(h5[self.fc_key][source_index])

        if fc_np.shape != (self.n_rois, self.n_rois):
            raise RuntimeError(
                f"Malformed static FC shape {fc_np.shape} "
                f"for source index {source_index}."
            )
        if not np.issubdtype(fc_np.dtype, np.floating):
            raise TypeError(
                f"Static FC must be floating point, found {fc_np.dtype}."
            )
        if not np.isfinite(fc_np).all():
            raise ValueError(
                f"Static FC contains NaN/Inf for "
                f"{self._all_subject_ids[source_index]}."
            )

        return StaticSample(
            subject_id=self._all_subject_ids[source_index],
            label=int(self._all_labels[source_index]),
            fc=torch.as_tensor(fc_np).to(dtype=self.dtype),
            source_index=source_index,
        )

    def source_group_name(self, index: int) -> str | None:
        """
        Return the original hierarchical HDF5 group name for provenance.
        Dense-layout files return None.
        """
        if self.layout != "subject_groups":
            return None
        assert self._source_groups is not None
        source_index = int(self._indices[index])
        return self._source_groups[source_index]

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
