from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from unified_bn.data.common import (
    canonical_subject_id, ensure_unique_subject_ids, index_subject_ids,
)

@dataclass(frozen=True)
class FrozenSplit:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    @property
    def all_ids(self):
        return self.train_ids + self.val_ids + self.test_ids

_ID_KEYS = {
    "train": ("train_ids", "train_subject_ids", "train_subjects"),
    "val": ("val_ids", "valid_ids", "validation_ids", "val_subject_ids"),
    "test": ("test_ids", "test_subject_ids", "test_subjects"),
}
_INDEX_KEYS = {
    "train": ("train_idx", "train_indices", "idx_train"),
    "val": ("val_idx", "valid_idx", "validation_idx", "idx_val"),
    "test": ("test_idx", "test_indices", "idx_test"),
}

def _find_unique_key(files, aliases, role):
    matches = [k for k in aliases if k in files]
    if len(matches) > 1:
        raise ValueError(f"Multiple candidate keys for {role}: {matches}.")
    return matches[0] if matches else None

def _as_ids(values):
    ids = tuple(canonical_subject_id(v) for v in np.asarray(values).reshape(-1))
    ensure_unique_subject_ids(ids)
    return ids

def _indices_to_ids(values, reference_subject_ids, role):
    arr = np.asarray(values).reshape(-1)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{role} indices must be integer-valued.")
    idx = arr.astype(np.int64, copy=False)
    if np.any(idx < 0) or np.any(idx >= len(reference_subject_ids)):
        raise IndexError(f"{role} split contains an out-of-range index.")
    ids = tuple(reference_subject_ids[int(i)] for i in idx)
    ensure_unique_subject_ids(ids)
    return ids

def load_frozen_split(
    path: str | Path, *, reference_subject_ids: Sequence[str] | None = None,
    require_complete_reference: bool = True,
) -> FrozenSplit:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen split not found: {path}")
    if reference_subject_ids is not None:
        ensure_unique_subject_ids(reference_subject_ids)

    with np.load(path, allow_pickle=False) as npz:
        files = tuple(npz.files)
        resolved = {}
        for part in ("train", "val", "test"):
            id_key = _find_unique_key(files, _ID_KEYS[part], f"{part} IDs")
            idx_key = _find_unique_key(files, _INDEX_KEYS[part], f"{part} indices")
            if id_key is not None and idx_key is not None:
                raise ValueError(
                    f"Both ID and index arrays exist for {part}; ambiguous."
                )
            if id_key is not None:
                resolved[part] = _as_ids(npz[id_key])
            elif idx_key is not None:
                if reference_subject_ids is None:
                    raise ValueError(
                        f"{part} uses indices ({idx_key}) but no "
                        "reference_subject_ids were supplied."
                    )
                resolved[part] = _indices_to_ids(
                    npz[idx_key], reference_subject_ids, part
                )
            else:
                raise KeyError(f"Could not find {part} split array. Keys: {files}")

    split = FrozenSplit(
        resolved["train"], resolved["val"], resolved["test"]
    )
    validate_frozen_split(
        split, reference_subject_ids=reference_subject_ids,
        require_complete_reference=require_complete_reference,
    )
    return split

def validate_frozen_split(
    split: FrozenSplit, *, reference_subject_ids=None,
    require_complete_reference=True,
) -> None:
    for name, ids in (
        ("train", split.train_ids), ("val", split.val_ids), ("test", split.test_ids)
    ):
        if not ids:
            raise ValueError(f"{name} split is empty.")
        ensure_unique_subject_ids(ids)

    train, val, test = set(split.train_ids), set(split.val_ids), set(split.test_ids)
    overlaps = {
        "train_val": train & val, "train_test": train & test, "val_test": val & test
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"Frozen split partitions overlap: {bad}")

    if reference_subject_ids is not None:
        reference = set(reference_subject_ids)
        split_set = train | val | test
        extra, missing = split_set - reference, reference - split_set
        if extra:
            raise ValueError(
                f"Split contains IDs absent from reference: {sorted(extra)[:10]}."
            )
        if require_complete_reference and missing:
            raise ValueError(
                f"Split does not cover all reference subjects: {sorted(missing)[:10]}."
            )

def split_indices(split: FrozenSplit, dataset_subject_ids: Sequence[str]):
    return {
        "train": index_subject_ids(dataset_subject_ids, split.train_ids),
        "val": index_subject_ids(dataset_subject_ids, split.val_ids),
        "test": index_subject_ids(dataset_subject_ids, split.test_ids),
    }
