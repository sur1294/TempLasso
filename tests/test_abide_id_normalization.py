import h5py
import numpy as np
import pytest
import torch

from unified_bn.data.abide_static import StaticABIDEDataset
from unified_bn.data.alignment import require_aligned_modalities


def _make_hierarchical(path):
    with h5py.File(path, "w") as h5:
        items = [
            ("sub-patient51585", 1, 3.0),
            ("sub-control50030", 0, 1.0),
            ("sub-patient50002", 1, 2.0),
        ]
        for group_name, _, scale in items:
            g = h5.create_group(f"{group_name}/schaefer100")
            g.create_dataset(
                "corr_matrix",
                data=np.eye(4, dtype=np.float32) * scale,
            )


def test_hierarchical_static_exposes_numeric_abide_ids(tmp_path):
    path = tmp_path / "static.h5"
    _make_hierarchical(path)

    ds = StaticABIDEDataset(path, expected_n_rois=4)

    assert ds.subject_ids == ("50030", "50002", "51585")
    assert ds.labels.tolist() == [0, 1, 1]

    assert ds.source_group_name(0) == "sub-control50030"
    assert ds.source_group_name(1) == "sub-patient50002"
    assert ds.source_group_name(2) == "sub-patient51585"

    assert ds[0].subject_id == "50030"
    assert ds[1].subject_id == "50002"
    assert ds[2].subject_id == "51585"


def test_hierarchical_static_subset_accepts_integer_split_ids(tmp_path):
    path = tmp_path / "static.h5"
    _make_hierarchical(path)

    ds = StaticABIDEDataset(
        path,
        expected_n_rois=4,
        subject_ids=[51585, np.int64(50030)],
    )

    assert ds.subject_ids == ("51585", "50030")
    assert ds.labels.tolist() == [1, 0]


def test_static_numeric_ids_align_with_temporal_numeric_ids(tmp_path):
    path = tmp_path / "static.h5"
    _make_hierarchical(path)

    ds = StaticABIDEDataset(path, expected_n_rois=4)

    temporal_ids = ("51585", "50002", "50030")
    temporal_labels = np.asarray([1, 1, 0], dtype=np.int64)

    report = require_aligned_modalities(
        ds.subject_ids,
        ds.labels,
        temporal_ids,
        temporal_labels,
    )

    assert report.exact_subject_match
    assert len(report.label_mismatches) == 0
    assert report.same_order is False


def test_unknown_static_group_prefix_is_not_silently_normalized(tmp_path):
    path = tmp_path / "bad.h5"

    with h5py.File(path, "w") as h5:
        g = h5.create_group("sub-other50030/schaefer100")
        g.create_dataset(
            "corr_matrix",
            data=np.eye(4, dtype=np.float32),
        )

    # No recognized hierarchical ABIDE layout is detected. The reader then
    # correctly fails trying to resolve a dense schema rather than inventing
    # semantics for the unknown group.
    with pytest.raises(KeyError):
        StaticABIDEDataset(path, expected_n_rois=4)
