from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from unified_bn.data import FusionABIDEDataset, FusionBatch, collate_fusion


def _edges(values):
    return np.stack(
        [np.full(6, value, dtype=np.float32) for value in values],
        axis=0,
    )


def _write_static(path, *, ids=("s3", "s1", "s2"), labels=(1, 0, 1)):
    fc = np.stack(
        [np.eye(4, dtype=np.float32) * (i + 1) for i in range(len(ids))]
    )
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(ids, dtype="S"))
        h5.create_dataset("labels", data=np.asarray(labels, dtype=np.int64))
        h5.create_dataset("static_fc", data=fc)


def _write_temporal(
    path,
    *,
    ids=("s2", "s3", "s1"),
    labels=(1, 1, 0),
    lengths=(3, 2, 1),
):
    values = []
    value = 0.1
    for length in lengths:
        for _ in range(length):
            values.append(value)
            value += 0.1

    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(ids, dtype="S"))
        h5.create_dataset("labels", data=np.asarray(labels, dtype=np.int64))
        h5.create_dataset("z_edges", data=_edges(values))
        h5.create_dataset("lengths", data=np.asarray(lengths, dtype=np.int64))


def _make_pair(tmp_path):
    static_path = tmp_path / "static.h5"
    temporal_path = tmp_path / "temporal.h5"
    _write_static(static_path)
    _write_temporal(temporal_path)
    return static_path, temporal_path


def test_fusion_dataset_aligns_by_subject_id_not_storage_position(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)

    ds = FusionABIDEDataset(
        static_path,
        temporal_path,
        expected_n_rois=4,
        subject_ids=["s1", "s2", "s3"],
    )

    assert ds.subject_ids == ("s1", "s2", "s3")
    assert ds.labels.tolist() == [0, 1, 1]
    assert ds.lengths.tolist() == [1, 3, 2]

    s1 = ds[0]
    assert s1.subject_id == "s1"
    assert s1.label == 0
    assert s1.static_source_index == 1
    assert s1.temporal_source_index == 2
    torch.testing.assert_close(s1.static_fc, torch.eye(4) * 2)
    assert s1.dynamic_fc.shape == (1, 4, 4)

    s2 = ds[1]
    assert s2.static_source_index == 2
    assert s2.temporal_source_index == 0
    assert s2.length == 3


def test_fusion_dataset_preserves_requested_frozen_split_order(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)

    ds = FusionABIDEDataset(
        static_path,
        temporal_path,
        expected_n_rois=4,
        subject_ids=["s2", "s1"],
    )

    assert ds.subject_ids == ("s2", "s1")
    assert [ds[i].subject_id for i in range(len(ds))] == ["s2", "s1"]


def test_fusion_dataset_default_order_is_static_source_order_but_id_aligned(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)

    ds = FusionABIDEDataset(
        static_path,
        temporal_path,
        expected_n_rois=4,
    )

    assert ds.subject_ids == ("s3", "s1", "s2")
    assert ds[0].subject_id == "s3"
    assert ds[0].static_source_index == 0
    assert ds[0].temporal_source_index == 1


def test_fusion_dataset_rejects_source_label_mismatch(tmp_path):
    static_path = tmp_path / "static.h5"
    temporal_path = tmp_path / "temporal.h5"
    _write_static(static_path)
    _write_temporal(temporal_path, labels=(1, 1, 1))

    with pytest.raises(ValueError, match="labels disagree"):
        FusionABIDEDataset(
            static_path,
            temporal_path,
            expected_n_rois=4,
        )


def test_fusion_dataset_rejects_different_subject_sets(tmp_path):
    static_path = tmp_path / "static.h5"
    temporal_path = tmp_path / "temporal.h5"
    _write_static(static_path)
    _write_temporal(
        temporal_path,
        ids=("s2", "s3", "s9"),
        labels=(1, 1, 0),
    )

    with pytest.raises(ValueError, match="subject sets differ"):
        FusionABIDEDataset(
            static_path,
            temporal_path,
            expected_n_rois=4,
        )


def test_fusion_dataset_rejects_missing_requested_subject(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)

    with pytest.raises(KeyError, match="requested IDs are absent"):
        FusionABIDEDataset(
            static_path,
            temporal_path,
            expected_n_rois=4,
            subject_ids=["s1", "missing"],
        )


def test_fusion_dataset_rejects_duplicate_requested_subjects(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)

    with pytest.raises(ValueError, match="Duplicate subject identifiers"):
        FusionABIDEDataset(
            static_path,
            temporal_path,
            expected_n_rois=4,
            subject_ids=["s1", "s1"],
        )


def test_fusion_collate_pads_temporal_only_and_preserves_provenance(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)
    ds = FusionABIDEDataset(
        static_path,
        temporal_path,
        expected_n_rois=4,
        subject_ids=["s1", "s2"],
    )

    batch = collate_fusion([ds[0], ds[1]])

    assert isinstance(batch, FusionBatch)
    assert batch.subject_ids == ("s1", "s2")
    assert torch.equal(batch.labels, torch.tensor([0, 1]))
    assert batch.static_fc.shape == (2, 4, 4)
    assert batch.dynamic_fc.shape == (2, 3, 4, 4)
    assert torch.equal(batch.lengths, torch.tensor([1, 3]))
    assert torch.equal(batch.dynamic_fc[0, 1:], torch.zeros(2, 4, 4))
    assert torch.equal(batch.static_source_indices, torch.tensor([1, 2]))
    assert torch.equal(batch.temporal_source_indices, torch.tensor([2, 0]))


def test_fusion_batch_to_preserves_ids_and_moves_all_tensor_fields(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)
    ds = FusionABIDEDataset(
        static_path,
        temporal_path,
        expected_n_rois=4,
        subject_ids=["s1", "s2"],
    )
    batch = collate_fusion([ds[0], ds[1]])

    moved = batch.to(torch.device("cpu"))

    assert moved.subject_ids == batch.subject_ids
    for tensor in (
        moved.labels,
        moved.static_fc,
        moved.dynamic_fc,
        moved.lengths,
        moved.static_source_indices,
        moved.temporal_source_indices,
    ):
        assert tensor.device.type == "cpu"


def test_fusion_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty fusion batch"):
        collate_fusion([])


def test_fusion_collate_rejects_static_temporal_roi_mismatch(tmp_path):
    static_path, temporal_path = _make_pair(tmp_path)
    ds = FusionABIDEDataset(
        static_path,
        temporal_path,
        expected_n_rois=4,
        subject_ids=["s1"],
    )
    sample = ds[0]
    bad = type(sample)(
        subject_id=sample.subject_id,
        label=sample.label,
        static_fc=sample.static_fc,
        dynamic_fc=torch.zeros(sample.length, 5, 5),
        length=sample.length,
        static_source_index=sample.static_source_index,
        temporal_source_index=sample.temporal_source_index,
    )

    with pytest.raises(ValueError, match="ROI shapes must match"):
        collate_fusion([bad])
