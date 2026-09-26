from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import h5py

def list_hdf5_datasets(handle: h5py.File) -> tuple[str, ...]:
    paths: list[str] = []
    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)
    handle.visititems(visitor)
    return tuple(sorted(paths))

def _normalize_h5_path(path: str) -> str:
    return path.strip().lstrip("/")

def resolve_dataset_path(
    handle: h5py.File, *, role: str, override: str | None,
    aliases: Sequence[str],
) -> str:
    all_paths = list_hdf5_datasets(handle)
    if override is not None:
        path = _normalize_h5_path(override)
        if path not in handle:
            raise KeyError(
                f"Configured {role} dataset {override!r} does not exist. "
                f"Available datasets: {all_paths}"
            )
        if not isinstance(handle[path], h5py.Dataset):
            raise TypeError(f"Configured {role} path {override!r} is not a dataset.")
        return path

    alias_set = {_normalize_h5_path(a) for a in aliases}
    exact = [p for p in all_paths if p in alias_set]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(
            f"Ambiguous HDF5 schema for {role}: exact matches {exact}. "
            "Provide an explicit schema override."
        )
    by_basename = [p for p in all_paths if p.rsplit("/", 1)[-1] in alias_set]
    if len(by_basename) == 1:
        return by_basename[0]
    if len(by_basename) > 1:
        raise RuntimeError(
            f"Ambiguous HDF5 schema for {role}: basename matches {by_basename}. "
            "Provide an explicit schema override."
        )
    raise KeyError(
        f"Could not resolve HDF5 dataset for {role}. Tried aliases "
        f"{tuple(aliases)}. Available datasets: {all_paths}"
    )

def resolve_optional_dataset_path(
    handle: h5py.File, *, role: str, override: str | None,
    aliases: Sequence[str],
) -> str | None:
    if override is not None:
        return resolve_dataset_path(
            handle, role=role, override=override, aliases=aliases
        )
    try:
        return resolve_dataset_path(
            handle, role=role, override=None, aliases=aliases
        )
    except KeyError:
        return None

@dataclass(frozen=True)
class StaticH5Schema:
    subject_ids: str | None = None
    labels: str | None = None
    fc: str | None = None
    atlas: str = "schaefer100"
    corr_dataset: str = "corr_matrix"

@dataclass(frozen=True)
class TemporalH5Schema:
    subject_ids: str | None = None
    labels: str | None = None
    edges: str | None = None
    lengths: str | None = None
    offsets: str | None = None
    triu_row: str | None = None
    triu_col: str | None = None

SUBJECT_ID_ALIASES = (
    "subject_ids", "subject_id", "participant_ids", "participant_id",
    "subjects", "ids",
)
LABEL_ALIASES = (
    "labels", "label", "targets", "target", "diagnosis", "dx", "y",
)
STATIC_FC_ALIASES = (
    "static_fc", "fc", "corr", "correlation", "connectivity",
    "connectivity_matrices", "corr_matrices",
)
TEMPORAL_EDGE_ALIASES = (
    "fc_z_upper",  # actual canonical temporal ABIDE artifact
    "temporal_fc", "dynamic_fc", "window_fc", "window_edges", "fc_edges",
    "z_edges", "fisher_z", "windows",
)
LENGTH_ALIASES = (
    "K",  # actual canonical temporal ABIDE artifact
    "lengths", "seq_lengths", "window_counts", "n_windows", "num_windows",
)
OFFSET_ALIASES = ("offsets", "window_offsets", "subject_offsets", "indptr")
TRIU_ROW_ALIASES = ("triu_row", "upper_row", "edge_row")
TRIU_COL_ALIASES = ("triu_col", "upper_col", "edge_col")
