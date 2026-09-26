import h5py
import numpy as np
import pytest

from unified_bn.data.abide_temporal import TemporalABIDEDataset


def _write_real_schema(path, *, corrupt_triu=False, corrupt_k=False):
    n_rois = 4
    rows, cols = np.triu_indices(n_rois, k=1)

    if corrupt_triu:
        rows = rows.copy()
        cols = cols.copy()
        rows[[0, 1]] = rows[[1, 0]]
        cols[[0, 1]] = cols[[1, 0]]

    K = np.asarray([2, 1, 3], dtype=np.int64)
    offsets = np.asarray([0, 2, 3, 6], dtype=np.int64)

    if corrupt_k:
        K = np.asarray([2, 2, 2], dtype=np.int64)

    edges = np.zeros((6, 6), dtype=np.float32)

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "subject_id",
            data=np.asarray(["s1", "s2", "s3"], dtype="S"),
        )
        h5.create_dataset("label", data=np.asarray([0, 1, 0]))
        h5.create_dataset("K", data=K)
        h5.create_dataset("offsets", data=offsets)
        h5.create_dataset("fc_z_upper", data=edges)
        h5.create_dataset("triu_row", data=rows)
        h5.create_dataset("triu_col", data=cols)
        # Extra provenance/QC datasets present in the actual artifact.
        h5.create_dataset("n_degenerate_rois", data=np.zeros(3, dtype=np.int64))
        h5.create_dataset("shrinkage", data=np.zeros(6, dtype=np.float32))
        h5.create_dataset("tr_native", data=np.full(3, 2.0))


def test_actual_temporal_schema_is_resolved(tmp_path):
    path = tmp_path / "real_temporal.h5"
    _write_real_schema(path)

    ds = TemporalABIDEDataset(path, expected_n_rois=4)

    assert ds.edge_key == "fc_z_upper"
    assert ds.length_key == "K"
    assert ds.offset_key == "offsets"
    assert ds.triu_row_key == "triu_row"
    assert ds.triu_col_key == "triu_col"
    assert ds.layout == "flat_lengths"
    assert ds.lengths.tolist() == [2, 1, 3]
    assert ds[2].dynamic_fc.shape == (3, 4, 4)


def test_actual_temporal_schema_rejects_k_offset_disagreement(tmp_path):
    path = tmp_path / "bad_k.h5"
    _write_real_schema(path, corrupt_k=True)

    with pytest.raises(ValueError, match="K/lengths and offsets disagree"):
        TemporalABIDEDataset(path, expected_n_rois=4)


def test_actual_temporal_schema_rejects_wrong_edge_order(tmp_path):
    path = tmp_path / "bad_triu.h5"
    _write_real_schema(path, corrupt_triu=True)

    with pytest.raises(ValueError, match="ordering does not match"):
        TemporalABIDEDataset(path, expected_n_rois=4)
