import h5py
import numpy as np
import torch

from unified_bn.data.abide_static import StaticABIDEDataset


def test_hierarchical_static_layout(tmp_path):
    path = tmp_path / "legacy_static.h5"

    with h5py.File(path, "w") as h5:
        for sid, scale in (
            ("sub-patient002", 2.0),
            ("sub-control001", 1.0),
        ):
            group = h5.create_group(f"{sid}/schaefer100")
            group.create_dataset(
                "corr_matrix",
                data=np.eye(4, dtype=np.float32) * scale,
            )
            group.create_dataset(
                "timeseries",
                data=np.zeros((20, 4), dtype=np.float32),
            )

    ds = StaticABIDEDataset(path, expected_n_rois=4)

    assert ds.layout == "subject_groups"
    assert ds.subject_ids == ("001", "002")
    assert ds.labels.tolist() == [0, 1]

    assert ds.source_group_name(0) == "sub-control001"
    assert ds.source_group_name(1) == "sub-patient002"

    s0 = ds[0]
    s1 = ds[1]

    assert s0.subject_id == "001"
    assert s0.label == 0
    assert s1.subject_id == "002"
    assert s1.label == 1

    torch.testing.assert_close(s0.fc, torch.eye(4))
    torch.testing.assert_close(s1.fc, torch.eye(4) * 2)


def test_hierarchical_layout_respects_subject_subset(tmp_path):
    path = tmp_path / "legacy_static.h5"

    with h5py.File(path, "w") as h5:
        for sid in (
            "sub-control001",
            "sub-patient002",
            "sub-control003",
        ):
            group = h5.create_group(f"{sid}/schaefer100")
            group.create_dataset(
                "corr_matrix",
                data=np.eye(4, dtype=np.float32),
            )

    ds = StaticABIDEDataset(
        path,
        expected_n_rois=4,
        subject_ids=["002", "001"],
    )

    assert ds.subject_ids == ("002", "001")
    assert ds.labels.tolist() == [1, 0]

    assert ds.source_group_name(0) == "sub-patient002"
    assert ds.source_group_name(1) == "sub-control001"