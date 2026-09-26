from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json

import h5py
import numpy as np
import torch

from unified_bn.data.abide_static import StaticABIDEDataset
from unified_bn.data.abide_temporal import TemporalABIDEDataset
from unified_bn.data.alignment import require_aligned_modalities
from unified_bn.data.common import fisher_z_upper_to_correlation
from unified_bn.data.schema import StaticH5Schema, TemporalH5Schema, list_hdf5_datasets
from unified_bn.data.splits import FrozenSplit, load_frozen_split, split_indices


@dataclass(frozen=True)
class StaticAuditSummary:
    n_subjects: int
    n_rois: int
    class_counts: dict[int, int]
    finite: bool
    symmetric_max_abs_error: float
    diagonal_min: float
    diagonal_max: float
    value_min: float
    value_max: float


@dataclass(frozen=True)
class TemporalAuditSummary:
    n_subjects: int
    n_rois: int
    n_edges: int
    n_windows: int
    min_windows: int
    max_windows: int
    mean_windows: float
    class_counts: dict[int, int]
    finite_raw_edges: bool
    reconstructed_symmetric_max_abs_error: float
    reconstructed_diagonal_min: float
    reconstructed_diagonal_max: float
    reconstructed_value_min: float
    reconstructed_value_max: float


@dataclass(frozen=True)
class AlignmentAuditSummary:
    n_common: int
    static_only: int
    temporal_only: int
    label_mismatches: int
    same_order: bool


@dataclass(frozen=True)
class SplitAuditSummary:
    file: str
    train_n: int
    val_n: int
    test_n: int
    total_n: int
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    static_resolves: bool
    temporal_resolves: bool


@dataclass(frozen=True)
class ReconstructionCheck:
    subject_id: str
    window_index: int
    max_abs_error: float
    symmetric_max_abs_error: float
    diagonal_max_abs_error: float


@dataclass(frozen=True)
class ABIDEAuditReport:
    static_datasets: tuple[str, ...]
    temporal_datasets: tuple[str, ...]
    static: StaticAuditSummary
    temporal: TemporalAuditSummary
    alignment: AlignmentAuditSummary
    splits: tuple[SplitAuditSummary, ...]
    reconstruction_checks: tuple[ReconstructionCheck, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _class_counts(labels: Sequence[int]) -> dict[int, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _safe_max(values: Iterable[float]) -> float:
    values = list(values)
    return float(max(values)) if values else 0.0


def _audit_static_full(
    dataset: StaticABIDEDataset,
    *,
    atol_symmetry: float,
) -> StaticAuditSummary:
    finite = True
    sym_errors: list[float] = []
    diag_min = float("inf")
    diag_max = float("-inf")
    value_min = float("inf")
    value_max = float("-inf")

    for i in range(len(dataset)):
        sample = dataset[i]
        fc = sample.fc

        finite = finite and bool(torch.isfinite(fc).all().item())
        sym_errors.append(float((fc - fc.T).abs().max().item()))
        diag = torch.diagonal(fc)
        diag_min = min(diag_min, float(diag.min().item()))
        diag_max = max(diag_max, float(diag.max().item()))
        value_min = min(value_min, float(fc.min().item()))
        value_max = max(value_max, float(fc.max().item()))

    max_sym = _safe_max(sym_errors)
    if not finite:
        raise AssertionError("Static FC contains non-finite values.")
    if max_sym > atol_symmetry:
        raise AssertionError(
            f"Static FC symmetry violation: max |A-A^T|={max_sym:.3e} "
            f"> tolerance {atol_symmetry:.3e}."
        )

    return StaticAuditSummary(
        n_subjects=len(dataset),
        n_rois=dataset.n_rois,
        class_counts=_class_counts(dataset.labels),
        finite=finite,
        symmetric_max_abs_error=max_sym,
        diagonal_min=diag_min,
        diagonal_max=diag_max,
        value_min=value_min,
        value_max=value_max,
    )


def _read_raw_temporal_edge_block(
    dataset: TemporalABIDEDataset,
    source_index: int,
) -> np.ndarray:
    """
    Read the raw Fisher-z edge block for one subject using dataset metadata.
    This deliberately bypasses __getitem__ reconstruction.
    """
    length = int(dataset._all_lengths[source_index])

    with h5py.File(dataset.path, "r") as h5:
        edge_ds = h5[dataset.edge_key]
        if dataset.layout == "padded":
            edges = np.asarray(edge_ds[source_index, :length, :])
        else:
            start = int(dataset._offsets[source_index])
            stop = int(dataset._offsets[source_index + 1])
            edges = np.asarray(edge_ds[start:stop, :])

    return edges


def _audit_temporal_full(
    dataset: TemporalABIDEDataset,
    *,
    atol_symmetry: float,
    atol_diagonal: float,
) -> TemporalAuditSummary:
    raw_finite = True
    sym_errors: list[float] = []
    diag_min = float("inf")
    diag_max = float("-inf")
    value_min = float("inf")
    value_max = float("-inf")

    for i in range(len(dataset)):
        source_index = int(dataset._indices[i])
        raw = _read_raw_temporal_edge_block(dataset, source_index)
        raw_finite = raw_finite and bool(np.isfinite(raw).all())

        sample = dataset[i]
        fc = sample.dynamic_fc

        sym_errors.append(
            float((fc - fc.transpose(-1, -2)).abs().max().item())
        )
        diag = torch.diagonal(fc, dim1=-2, dim2=-1)
        diag_min = min(diag_min, float(diag.min().item()))
        diag_max = max(diag_max, float(diag.max().item()))
        value_min = min(value_min, float(fc.min().item()))
        value_max = max(value_max, float(fc.max().item()))

    max_sym = _safe_max(sym_errors)
    if not raw_finite:
        raise AssertionError("Temporal raw Fisher-z edges contain non-finite values.")
    if max_sym > atol_symmetry:
        raise AssertionError(
            f"Temporal reconstructed FC symmetry violation: "
            f"max |A-A^T|={max_sym:.3e} > {atol_symmetry:.3e}."
        )
    if abs(diag_min - 1.0) > atol_diagonal or abs(diag_max - 1.0) > atol_diagonal:
        raise AssertionError(
            "Temporal reconstructed diagonal is not exactly/approximately 1: "
            f"min={diag_min:.9f}, max={diag_max:.9f}."
        )
    if value_min < -1.0 - 1e-6 or value_max > 1.0 + 1e-6:
        raise AssertionError(
            f"Temporal reconstructed correlations outside [-1,1]: "
            f"[{value_min}, {value_max}]."
        )

    return TemporalAuditSummary(
        n_subjects=len(dataset),
        n_rois=dataset.n_rois,
        n_edges=dataset.n_edges,
        n_windows=int(dataset.lengths.sum()),
        min_windows=dataset.min_windows,
        max_windows=dataset.max_windows,
        mean_windows=dataset.mean_windows,
        class_counts=_class_counts(dataset.labels),
        finite_raw_edges=raw_finite,
        reconstructed_symmetric_max_abs_error=max_sym,
        reconstructed_diagonal_min=diag_min,
        reconstructed_diagonal_max=diag_max,
        reconstructed_value_min=value_min,
        reconstructed_value_max=value_max,
    )


def _audit_reconstruction_independently(
    dataset: TemporalABIDEDataset,
    *,
    n_subjects: int = 3,
    windows_per_subject: int = 2,
) -> tuple[ReconstructionCheck, ...]:
    """
    Independently reproduce selected raw-edge -> dense-correlation conversions.

    This compares the dataset's returned matrices against a fresh direct
    reconstruction from the raw HDF5 edge rows.
    """
    if n_subjects <= 0 or windows_per_subject <= 0:
        raise ValueError("n_subjects and windows_per_subject must be positive.")

    n_pick = min(n_subjects, len(dataset))
    positions = np.linspace(0, len(dataset) - 1, num=n_pick, dtype=int)
    checks: list[ReconstructionCheck] = []

    for dataset_index in positions:
        source_index = int(dataset._indices[int(dataset_index)])
        sample = dataset[int(dataset_index)]
        raw = _read_raw_temporal_edge_block(dataset, source_index)

        k = sample.length
        if windows_per_subject >= k:
            window_indices = np.arange(k, dtype=int)
        else:
            window_indices = np.linspace(
                0, k - 1, num=windows_per_subject, dtype=int
            )

        for w in window_indices:
            manual = fisher_z_upper_to_correlation(
                raw[int(w)],
                n_rois=dataset.n_rois,
                dtype=sample.dynamic_fc.dtype,
            )
            loaded = sample.dynamic_fc[int(w)]

            max_abs_error = float((manual - loaded).abs().max().item())
            sym_error = float((loaded - loaded.T).abs().max().item())
            diag_error = float(
                (torch.diagonal(loaded) - 1.0).abs().max().item()
            )

            checks.append(
                ReconstructionCheck(
                    subject_id=sample.subject_id,
                    window_index=int(w),
                    max_abs_error=max_abs_error,
                    symmetric_max_abs_error=sym_error,
                    diagonal_max_abs_error=diag_error,
                )
            )

    return tuple(checks)


def _audit_splits(
    split_paths: Sequence[Path],
    static: StaticABIDEDataset,
    temporal: TemporalABIDEDataset,
) -> tuple[SplitAuditSummary, ...]:
    out: list[SplitAuditSummary] = []

    for path in split_paths:
        # ID-based split files are preferred. If the actual file is index-based,
        # the static subject order becomes the explicit reference order.
        try:
            split = load_frozen_split(
                path,
                reference_subject_ids=static.subject_ids,
                require_complete_reference=True,
            )
        except ValueError as exc:
            # Preserve fail-closed behavior. The fallback is only for the known
            # case where the NPZ stores subject IDs and reference conversion is
            # unnecessary; other validation failures still propagate.
            if "reference_subject_ids" in str(exc):
                raise
            raise

        train, val, test = (
            set(split.train_ids),
            set(split.val_ids),
            set(split.test_ids),
        )

        static_resolves = True
        temporal_resolves = True
        try:
            split_indices(split, static.subject_ids)
        except Exception:
            static_resolves = False
        try:
            split_indices(split, temporal.subject_ids)
        except Exception:
            temporal_resolves = False

        if not static_resolves or not temporal_resolves:
            raise AssertionError(
                f"Frozen split {path.name} does not resolve in both modalities."
            )

        out.append(
            SplitAuditSummary(
                file=str(path),
                train_n=len(split.train_ids),
                val_n=len(split.val_ids),
                test_n=len(split.test_ids),
                total_n=len(split.all_ids),
                train_val_overlap=len(train & val),
                train_test_overlap=len(train & test),
                val_test_overlap=len(val & test),
                static_resolves=static_resolves,
                temporal_resolves=temporal_resolves,
            )
        )

    return tuple(out)


def run_abide_audit(
    *,
    static_h5: str | Path,
    temporal_h5: str | Path,
    split_paths: Sequence[str | Path],
    static_schema: StaticH5Schema | None = None,
    temporal_schema: TemporalH5Schema | None = None,
    expected_n_rois: int = 100,
    expected_n_subjects: int | None = 1025,
    expected_n_edges: int | None = 4950,
    expected_total_windows: int | None = 17551,
    expected_min_windows: int | None = 12,
    expected_max_windows: int | None = 26,
    expected_class_counts: dict[int, int] | None = None,
    expected_split_sizes: tuple[int, int, int] | None = (717, 103, 205),
    atol_symmetry: float = 1e-6,
    atol_diagonal: float = 1e-7,
) -> ABIDEAuditReport:
    """
    Perform an end-to-end scientific audit of the actual ABIDE static,
    temporal, alignment, and frozen-split artifacts.

    Expected values are assertions, not transformations. Set any optional
    expected value to None when intentionally auditing a different cohort.
    """
    static_h5 = Path(static_h5)
    temporal_h5 = Path(temporal_h5)
    split_paths = tuple(Path(p) for p in split_paths)

    if not split_paths:
        raise ValueError("At least one frozen split file must be supplied.")

    with h5py.File(static_h5, "r") as h5:
        static_datasets = list_hdf5_datasets(h5)
    with h5py.File(temporal_h5, "r") as h5:
        temporal_datasets = list_hdf5_datasets(h5)

    static = StaticABIDEDataset(
        static_h5,
        schema=static_schema,
        expected_n_rois=expected_n_rois,
        dtype=torch.float32,
    )
    temporal = TemporalABIDEDataset(
        temporal_h5,
        schema=temporal_schema,
        expected_n_rois=expected_n_rois,
        dtype=torch.float32,
    )

    try:
        static_summary = _audit_static_full(
            static,
            atol_symmetry=atol_symmetry,
        )
        temporal_summary = _audit_temporal_full(
            temporal,
            atol_symmetry=atol_symmetry,
            atol_diagonal=atol_diagonal,
        )

        alignment_report = require_aligned_modalities(
            static.subject_ids,
            static.labels,
            temporal.subject_ids,
            temporal.labels,
        )
        alignment_summary = AlignmentAuditSummary(
            n_common=alignment_report.n_common,
            static_only=len(alignment_report.left_only),
            temporal_only=len(alignment_report.right_only),
            label_mismatches=len(alignment_report.label_mismatches),
            same_order=alignment_report.same_order,
        )

        split_summaries = _audit_splits(split_paths, static, temporal)
        reconstruction_checks = _audit_reconstruction_independently(temporal)

        if expected_n_subjects is not None:
            assert static_summary.n_subjects == expected_n_subjects, (
                f"Static N={static_summary.n_subjects}, "
                f"expected {expected_n_subjects}."
            )
            assert temporal_summary.n_subjects == expected_n_subjects, (
                f"Temporal N={temporal_summary.n_subjects}, "
                f"expected {expected_n_subjects}."
            )

        if expected_n_edges is not None:
            assert temporal_summary.n_edges == expected_n_edges, (
                f"Temporal E={temporal_summary.n_edges}, expected {expected_n_edges}."
            )

        if expected_total_windows is not None:
            assert temporal_summary.n_windows == expected_total_windows, (
                f"Temporal total windows={temporal_summary.n_windows}, "
                f"expected {expected_total_windows}."
            )

        if expected_min_windows is not None:
            assert temporal_summary.min_windows == expected_min_windows, (
                f"K_min={temporal_summary.min_windows}, "
                f"expected {expected_min_windows}."
            )

        if expected_max_windows is not None:
            assert temporal_summary.max_windows == expected_max_windows, (
                f"K_max={temporal_summary.max_windows}, "
                f"expected {expected_max_windows}."
            )

        if expected_class_counts is not None:
            assert static_summary.class_counts == expected_class_counts, (
                f"Static class counts={static_summary.class_counts}, "
                f"expected={expected_class_counts}."
            )
            assert temporal_summary.class_counts == expected_class_counts, (
                f"Temporal class counts={temporal_summary.class_counts}, "
                f"expected={expected_class_counts}."
            )

        if expected_split_sizes is not None:
            expected_train, expected_val, expected_test = expected_split_sizes
            for summary in split_summaries:
                assert summary.train_n == expected_train
                assert summary.val_n == expected_val
                assert summary.test_n == expected_test
                assert summary.total_n == sum(expected_split_sizes)
                assert summary.train_val_overlap == 0
                assert summary.train_test_overlap == 0
                assert summary.val_test_overlap == 0

        for check in reconstruction_checks:
            assert check.max_abs_error == 0.0, (
                f"Manual reconstruction mismatch for {check.subject_id}, "
                f"window {check.window_index}: {check.max_abs_error:.3e}"
            )

        return ABIDEAuditReport(
            static_datasets=static_datasets,
            temporal_datasets=temporal_datasets,
            static=static_summary,
            temporal=temporal_summary,
            alignment=alignment_summary,
            splits=split_summaries,
            reconstruction_checks=reconstruction_checks,
        )
    finally:
        static.close()
        temporal.close()
