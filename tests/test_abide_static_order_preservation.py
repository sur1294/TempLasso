import h5py
import numpy as np

from unified_bn.data.abide_static import StaticABIDEDataset


def test_canonicalization_does_not_reorder_hierarchical_static_subjects(tmp_path):
    path = tmp_path / "static_order.h5"

    with h5py.File(path, "w") as h5:
        for group_name in (
            "sub-patient002",
            "sub-control003",
            "sub-control001",
        ):
            g = h5.create_group(f"{group_name}/schaefer100")
            g.create_dataset(
                "corr_matrix",
                data=np.eye(4, dtype=np.float32),
            )

    ds = StaticABIDEDataset(path, expected_n_rois=4)

    assert ds.subject_ids == ("001", "003", "002")
    assert ds.source_group_name(0) == "sub-control001"
    assert ds.source_group_name(1) == "sub-control003"
    assert ds.source_group_name(2) == "sub-patient002"
    assert ds.labels.tolist() == [0, 0, 1]
