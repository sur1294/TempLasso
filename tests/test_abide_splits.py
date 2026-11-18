import numpy as np
import pytest
from unified_bn.data.splits import load_frozen_split, split_indices

def test_load_id_based_frozen_split(tmp_path):
    path = tmp_path / "split.npz"
    np.savez(
        path,
        train_ids=np.asarray(["b","a"]),
        val_ids=np.asarray(["c"]),
        test_ids=np.asarray(["d"]),
    )
    split = load_frozen_split(
        path, reference_subject_ids=["a","b","c","d"]
    )
    assert split.train_ids == ("b","a")
    idx = split_indices(split, ["a","b","c","d"])
    assert idx["train"].tolist() == [1,0]

def test_index_split_requires_reference_order(tmp_path):
    path = tmp_path / "split_idx.npz"
    np.savez(
        path,
        train_idx=np.asarray([1,0]),
        val_idx=np.asarray([2]),
        test_idx=np.asarray([3]),
    )
    with pytest.raises(ValueError, match="reference_subject_ids"):
        load_frozen_split(path)
    split = load_frozen_split(
        path, reference_subject_ids=["a","b","c","d"]
    )
    assert split.train_ids == ("b","a")

def test_split_overlap_rejected(tmp_path):
    path = tmp_path / "overlap.npz"
    np.savez(
        path,
        train_ids=np.asarray(["a","b"]),
        val_ids=np.asarray(["b"]),
        test_ids=np.asarray(["c"]),
    )
    with pytest.raises(ValueError, match="overlap"):
        load_frozen_split(path, reference_subject_ids=["a","b","c"])

def test_split_missing_reference_rejected(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(
        path,
        train_ids=np.asarray(["a"]),
        val_ids=np.asarray(["b"]),
        test_ids=np.asarray(["c"]),
    )
    with pytest.raises(ValueError, match="does not cover"):
        load_frozen_split(
            path, reference_subject_ids=["a","b","c","d"]
        )
