from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch

def canonical_subject_id(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    elif isinstance(value, (str, int, np.integer)):
        text = str(value)
    elif isinstance(value, float) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value)
    text = text.strip()
    if not text:
        raise ValueError("Encountered an empty subject identifier.")
    return text

def canonicalize_subject_ids(values: Sequence[object]) -> tuple[str, ...]:
    ids = tuple(canonical_subject_id(v) for v in values)
    ensure_unique_subject_ids(ids)
    return ids

def ensure_unique_subject_ids(subject_ids: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for sid in subject_ids:
        if sid in seen:
            duplicates.append(sid)
        seen.add(sid)
    if duplicates:
        dupes = sorted(set(duplicates))
        raise ValueError(
            f"Duplicate subject identifiers are not allowed: {dupes[:10]}"
        )

def normalize_labels(values: np.ndarray, *, num_classes: int | None = None) -> np.ndarray:
    """
    Validate and return integer class labels.

    ``num_classes=None`` (the historical contract) requires binary ``{0,1}``
    labels. Otherwise labels must lie in ``{0, ..., num_classes-1}``.
    """
    if num_classes is not None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        raise ValueError("Label array is empty.")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError("Labels contain NaN or Inf.")
        rounded = np.rint(arr)
        if not np.array_equal(arr, rounded):
            raise ValueError("Floating labels contain non-integral values.")
        arr = rounded
    try:
        arr = arr.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("Labels must be integer-convertible.") from exc
    unique = set(np.unique(arr).tolist())
    if num_classes is None:
        if not unique.issubset({0, 1}):
            raise ValueError(f"Expected binary labels {{0,1}}, found {sorted(unique)}.")
    elif not unique.issubset(set(range(num_classes))):
        raise ValueError(
            f"Expected labels in 0..{num_classes - 1}, found {sorted(unique)}."
        )
    return arr


def normalize_binary_labels(values: np.ndarray) -> np.ndarray:
    return normalize_labels(values)

def infer_n_rois_from_upper_edges(n_edges: int) -> int:
    if n_edges <= 0:
        raise ValueError("n_edges must be positive.")
    discriminant = 1 + 8 * n_edges
    root = int(np.sqrt(discriminant))
    if root * root != discriminant:
        raise ValueError(
            f"Edge count {n_edges} is not a triangular number R(R-1)/2."
        )
    n_rois = (1 + root) // 2
    if n_rois * (n_rois - 1) // 2 != n_edges:
        raise ValueError("Edge count does not correspond to an integer ROI count.")
    return n_rois

def fisher_z_upper_to_correlation(
    z_upper: np.ndarray | torch.Tensor, *, n_rois: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    z = torch.as_tensor(z_upper)
    if z.ndim < 1:
        raise ValueError("z_upper must have at least one dimension.")
    if not torch.is_floating_point(z):
        z = z.to(torch.float32)
    n_edges = int(z.shape[-1])
    inferred = infer_n_rois_from_upper_edges(n_edges)
    if n_rois is None:
        n_rois = inferred
    elif int(n_rois) != inferred:
        raise ValueError(
            f"Configured n_rois={n_rois} conflicts with E={n_edges}, "
            f"which implies n_rois={inferred}."
        )
    if not torch.isfinite(z).all():
        raise ValueError("Temporal Fisher-z edges contain NaN or Inf.")
    z = z.to(dtype=dtype)
    dense_z = torch.zeros(
        (*z.shape[:-1], n_rois, n_rois), dtype=dtype, device=z.device
    )
    rows, cols = torch.triu_indices(n_rois, n_rois, offset=1, device=z.device)
    dense_z[..., rows, cols] = z
    dense_z[..., cols, rows] = z
    corr = torch.tanh(dense_z)
    diag = torch.arange(n_rois, device=z.device)
    corr[..., diag, diag] = 1.0
    return corr

def index_subject_ids(
    all_subject_ids: Sequence[str], requested_subject_ids: Sequence[str],
) -> np.ndarray:
    ensure_unique_subject_ids(all_subject_ids)
    ensure_unique_subject_ids(requested_subject_ids)
    lookup = {sid: idx for idx, sid in enumerate(all_subject_ids)}
    missing = [sid for sid in requested_subject_ids if sid not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} requested IDs are absent: {missing[:10]}"
        )
    return np.asarray([lookup[sid] for sid in requested_subject_ids], dtype=np.int64)

@dataclass(frozen=True)
class StaticSample:
    subject_id: str
    label: int
    fc: torch.Tensor
    source_index: int

@dataclass(frozen=True)
class TemporalSample:
    subject_id: str
    label: int
    dynamic_fc: torch.Tensor
    length: int
    source_index: int
