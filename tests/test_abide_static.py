import h5py
import numpy as np
import pytest
import torch
from unified_bn.data.abide_static import StaticABIDEDataset
from unified_bn.data.collate import collate_static

def _write_static(path):
    ids = ["s3", "s1", "s2"]
    labels = np.array([1, 0, 1], dtype=np.int64)
    fc = np.stack([np.eye(4, dtype=np.float32) * (i + 1) for i in range(3)])
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(ids, dtype="S"))
        h5.create_dataset("labels", data=labels)
        h5.create_dataset("static_fc", data=fc)

def test_static_dataset_preserves_raw_matrix(tmp_path):
    path = tmp_path / "static.h5"
    _write_static(path)
    ds = StaticABIDEDataset(path, expected_n_rois=4)
    sample = ds[1]
    assert sample.subject_id == "s1"
    assert sample.label == 0
    torch.testing.assert_close(sample.fc, torch.eye(4) * 2)

def test_static_subset_resolves_by_subject_id(tmp_path):
    path = tmp_path / "static.h5"
    _write_static(path)
    ds = StaticABIDEDataset(
        path, expected_n_rois=4, subject_ids=["s2", "s3"]
    )
    assert ds.subject_ids == ("s2", "s3")
    assert ds[0].source_index == 2
    assert ds[1].source_index == 0

def test_static_nan_rejected_at_read_time(tmp_path):
    path = tmp_path / "static.h5"
    _write_static(path)
    with h5py.File(path, "r+") as h5:
        h5["static_fc"][0, 1, 2] = np.nan
    ds = StaticABIDEDataset(path, expected_n_rois=4)
    with pytest.raises(ValueError, match="NaN/Inf"):
        _ = ds[0]

def test_static_collate(tmp_path):
    path = tmp_path / "static.h5"
    _write_static(path)
    ds = StaticABIDEDataset(path, expected_n_rois=4)
    batch = collate_static([ds[0], ds[2]])
    assert batch.subject_ids == ("s3", "s2")
    assert batch.fc.shape == (2, 4, 4)
    assert torch.equal(batch.labels, torch.tensor([1, 1]))
