import h5py
import numpy as np
import pytest
import torch
from unified_bn.data.abide_temporal import TemporalABIDEDataset
from unified_bn.data.collate import collate_temporal
from unified_bn.data.common import fisher_z_upper_to_correlation

def _edges(values):
    return np.stack(
        [np.full(6, v, dtype=np.float32) for v in values], axis=0
    )

def _write_flat_lengths(path):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(["a","b","c"], dtype="S"))
        h5.create_dataset("labels", data=np.asarray([0,1,0]))
        h5.create_dataset("z_edges", data=_edges([0,.1,.2,.3,.4,.5]))
        h5.create_dataset("lengths", data=np.asarray([2,1,3]))

def test_fisher_z_reconstruction_properties():
    z = torch.tensor([[0.0, 0.1, -0.2, 0.3, -0.4, 0.5]])
    corr = fisher_z_upper_to_correlation(z, n_rois=4)
    assert corr.shape == (1,4,4)
    torch.testing.assert_close(corr, corr.transpose(-1,-2))
    torch.testing.assert_close(
        torch.diagonal(corr, dim1=-2, dim2=-1), torch.ones(1,4)
    )
    assert torch.isclose(corr[0,0,2], torch.tanh(torch.tensor(0.1)))

def test_temporal_flat_lengths_preserves_native_k(tmp_path):
    path = tmp_path / "temporal.h5"
    _write_flat_lengths(path)
    ds = TemporalABIDEDataset(path, expected_n_rois=4)
    assert ds.layout == "flat_lengths"
    assert ds.lengths.tolist() == [2,1,3]
    assert ds.min_windows == 1
    assert ds.max_windows == 3
    assert ds[2].dynamic_fc.shape == (3,4,4)

def test_temporal_padded_layout_ignores_padding(tmp_path):
    path = tmp_path / "padded.h5"
    padded = np.zeros((2,4,6), dtype=np.float32)
    padded[0,:2] = _edges([.1,.2])
    padded[1,0] = _edges([.3])[0]
    padded[:,2:] = 999.0
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(["x","y"], dtype="S"))
        h5.create_dataset("labels", data=np.asarray([0,1]))
        h5.create_dataset("temporal_fc", data=padded)
        h5.create_dataset("lengths", data=np.asarray([2,1]))
    ds = TemporalABIDEDataset(path, expected_n_rois=4)
    assert ds.layout == "padded"
    assert ds[0].dynamic_fc.shape[0] == 2
    assert ds[1].dynamic_fc.shape[0] == 1

def test_temporal_offsets_layout(tmp_path):
    path = tmp_path / "offsets.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(["x","y"], dtype="S"))
        h5.create_dataset("labels", data=np.asarray([0,1]))
        h5.create_dataset("window_edges", data=_edges([.1,.2,.3]))
        h5.create_dataset("offsets", data=np.asarray([0,1,3]))
    ds = TemporalABIDEDataset(path, expected_n_rois=4)
    assert ds.layout == "flat_offsets"
    assert ds.lengths.tolist() == [1,2]

def test_temporal_length_sum_mismatch_rejected(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(["x","y"], dtype="S"))
        h5.create_dataset("labels", data=np.asarray([0,1]))
        h5.create_dataset("z_edges", data=np.zeros((3,6), dtype=np.float32))
        h5.create_dataset("lengths", data=np.asarray([1,1]))
    with pytest.raises(ValueError, match=r"sum\(lengths\)"):
        TemporalABIDEDataset(path, expected_n_rois=4)

def test_temporal_collate_preserves_lengths_and_pads(tmp_path):
    path = tmp_path / "temporal.h5"
    _write_flat_lengths(path)
    ds = TemporalABIDEDataset(path, expected_n_rois=4)
    batch = collate_temporal([ds[0], ds[2]])
    assert batch.dynamic_fc.shape == (2,3,4,4)
    assert torch.equal(batch.lengths, torch.tensor([2,3]))
    assert torch.equal(batch.dynamic_fc[0,2], torch.zeros(4,4))
