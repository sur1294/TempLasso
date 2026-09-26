from __future__ import annotations

import h5py
import numpy as np

from unified_bn.data.audit import run_abide_audit


def _make_static(path):
    ids = np.asarray(["s2", "s1", "s3", "s4"], dtype="S")
    labels = np.asarray([1, 0, 0, 1], dtype=np.int64)
    fc = np.stack(
        [np.eye(4, dtype=np.float32) for _ in range(4)],
        axis=0,
    )
    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=ids)
        h5.create_dataset("labels", data=labels)
        h5.create_dataset("static_fc", data=fc)


def _make_temporal(path):
    # Same subjects/labels, deliberately different row order.
    ids = np.asarray(["s1", "s4", "s2", "s3"], dtype="S")
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    lengths = np.asarray([2, 1, 3, 2], dtype=np.int64)
    # R=4 => E=6.
    rows = []
    for i in range(int(lengths.sum())):
        rows.append(np.full(6, 0.01 * i, dtype=np.float32))
    edges = np.stack(rows, axis=0)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=ids)
        h5.create_dataset("labels", data=labels)
        h5.create_dataset("z_edges", data=edges)
        h5.create_dataset("lengths", data=lengths)


def _make_split(path):
    np.savez(
        path,
        train_ids=np.asarray(["s1", "s2"]),
        val_ids=np.asarray(["s3"]),
        test_ids=np.asarray(["s4"]),
    )


def test_end_to_end_audit_on_synthetic_artifacts(tmp_path):
    static_h5 = tmp_path / "static.h5"
    temporal_h5 = tmp_path / "temporal.h5"
    split = tmp_path / "split.npz"

    _make_static(static_h5)
    _make_temporal(temporal_h5)
    _make_split(split)

    report = run_abide_audit(
        static_h5=static_h5,
        temporal_h5=temporal_h5,
        split_paths=[split],
        expected_n_rois=4,
        expected_n_subjects=4,
        expected_n_edges=6,
        expected_total_windows=8,
        expected_min_windows=1,
        expected_max_windows=3,
        expected_class_counts={0: 2, 1: 2},
        expected_split_sizes=(2, 1, 1),
    )

    assert report.static.n_subjects == 4
    assert report.temporal.n_windows == 8
    assert report.alignment.n_common == 4
    assert report.alignment.label_mismatches == 0
    assert report.alignment.same_order is False
    assert len(report.reconstruction_checks) > 0
    assert all(c.max_abs_error == 0.0 for c in report.reconstruction_checks)
