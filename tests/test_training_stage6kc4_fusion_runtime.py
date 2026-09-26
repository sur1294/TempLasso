from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

import unified_bn.training.abide_runtime as runtime_module
from unified_bn.backbones.bqn import BQNBackboneConfig
from unified_bn.data import FusionBatch, load_frozen_split
from unified_bn.temporal.bigru import BiGRUConfig
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.abide_runtime import (
    ABIDERuntimeConfig,
    ABIDERuntimeFactory,
    frozen_split_identity,
)
from unified_bn.training.config import CandidateRunConfig, RunIdentity, TrainingConfig


STATIC_SUBJECTS = ("s1", "s2", "s3", "s4")
STATIC_LABELS = np.asarray([0, 1, 0, 1], dtype=np.int64)

# Deliberately different storage order from static HDF5. Fusion must align by
# canonical subject ID rather than by row position.
TEMPORAL_SUBJECTS = ("s4", "s2", "s1", "s3")
TEMPORAL_LABELS = np.asarray([1, 1, 0, 0], dtype=np.int64)
TEMPORAL_LENGTHS = np.asarray([2, 3, 1, 2], dtype=np.int64)


def _write_split(path: Path) -> None:
    np.savez(
        path,
        train_ids=np.asarray(["s2", "s1"]),
        val_ids=np.asarray(["s3"]),
        test_ids=np.asarray(["s4"]),
        seed=np.asarray(42, dtype=np.int64),
    )


def _write_static_h5(path: Path) -> None:
    fc = np.zeros((4, 3, 3), dtype=np.float32)
    for i in range(4):
        fc[i] = np.eye(3, dtype=np.float32)
        fc[i, 0, 1] = fc[i, 1, 0] = 0.1 * (i + 1)

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "subject_ids",
            data=np.asarray(STATIC_SUBJECTS, dtype="S"),
        )
        h5.create_dataset("labels", data=STATIC_LABELS)
        h5.create_dataset("fc", data=fc)


def _write_temporal_h5(path: Path) -> None:
    offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(TEMPORAL_LENGTHS),
        ]
    )
    rows, cols = np.triu_indices(3, k=1)

    windows = int(TEMPORAL_LENGTHS.sum())
    edges = np.zeros((windows, len(rows)), dtype=np.float32)
    for i in range(windows):
        edges[i] = 0.01 * (i + 1)

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "subject_ids",
            data=np.asarray(TEMPORAL_SUBJECTS, dtype="S"),
        )
        h5.create_dataset("labels", data=TEMPORAL_LABELS)
        h5.create_dataset("K", data=TEMPORAL_LENGTHS)
        h5.create_dataset("offsets", data=offsets)
        h5.create_dataset("fc_z_upper", data=edges)
        h5.create_dataset("triu_row", data=rows.astype(np.int64))
        h5.create_dataset("triu_col", data=cols.astype(np.int64))


def _artifacts(tmp_path: Path):
    split_path = tmp_path / "split.npz"
    static_h5 = tmp_path / "static.h5"
    temporal_h5 = tmp_path / "temporal.h5"
    _write_split(split_path)
    _write_static_h5(static_h5)
    _write_temporal_h5(temporal_h5)
    return split_path, static_h5, temporal_h5


def _split_identity(split_path: Path) -> str:
    return frozen_split_identity(load_frozen_split(split_path))


def _fusion_candidate(split_path: Path) -> CandidateRunConfig:
    bqn = {
        "name": "bqn",
        **asdict(
            BQNBackboneConfig(
                n_rois=3,
                n_layers=1,
                dropout=0.0,
                activation="elu",
                cluster_num=2,
            )
        ),
    }

    bigru = asdict(
        BiGRUConfig(
            input_dim=32,
            hidden_dim=4,
            num_layers=1,
            dropout=0.0,
            pooling="final",
            projection_dim=None,
            layer_norm=False,
        )
    )
    bigru.pop("input_dim")
    bigru.pop("attention_hidden_dim")
    bigru.pop("attention_dropout")

    return CandidateRunConfig(
        identity=RunIdentity(
            mode="fusion",
            seed=42,
            split_identity=_split_identity(split_path),
            backbone_config=bqn,
            temporal_config={"name": "bigru", **bigru},
            window_policy=WindowPolicyConfig(policy="native"),
            fusion_config={"name": "concat"},
        ),
        training=TrainingConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
        ),
    )


def _factory(
    split_path: Path,
    static_h5: Path,
    temporal_h5: Path,
) -> ABIDERuntimeFactory:
    return ABIDERuntimeFactory(
        ABIDERuntimeConfig(
            split_path=split_path,
            static_h5=static_h5,
            temporal_h5=temporal_h5,
            device="cpu",
            expected_n_rois=3,
            pin_memory=False,
        )
    )


def test_fusion_runtime_builds_only_frozen_train_and_val_batches(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _fusion_candidate(split_path)

    sentinel_model = nn.Linear(1, 1)
    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: sentinel_model,
    )

    runtime = _factory(
        split_path,
        static_h5,
        temporal_h5,
    )(candidate, 0, tmp_path / "run")

    assert runtime.model is sentinel_model
    assert tuple(runtime.train_batches.dataset.subject_ids) == ("s2", "s1")
    assert tuple(runtime.val_batches.dataset.subject_ids) == ("s3",)
    assert not hasattr(runtime, "test_batches")

    batch = next(iter(runtime.train_batches))
    assert isinstance(batch, FusionBatch)
    assert set(batch.subject_ids) == {"s1", "s2"}
    assert batch.static_fc.shape == (2, 3, 3)
    assert batch.dynamic_fc.shape == (2, 3, 3, 3)


def test_fusion_runtime_aligns_reordered_sources_by_subject_id(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _fusion_candidate(split_path)

    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: nn.Linear(1, 1),
    )

    runtime = _factory(
        split_path,
        static_h5,
        temporal_h5,
    )(candidate, 0, tmp_path / "run")

    batch = next(iter(runtime.train_batches))
    observed = {
        subject_id: (
            int(batch.static_source_indices[i]),
            int(batch.temporal_source_indices[i]),
            int(batch.lengths[i]),
            int(batch.labels[i]),
        )
        for i, subject_id in enumerate(batch.subject_ids)
    }

    assert observed["s1"] == (0, 2, 1, 0)
    assert observed["s2"] == (1, 1, 3, 1)


def test_fusion_runtime_preserves_native_lengths_and_temporal_only_padding(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _fusion_candidate(split_path)

    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: nn.Linear(1, 1),
    )

    runtime = _factory(
        split_path,
        static_h5,
        temporal_h5,
    )(candidate, 0, tmp_path / "run")
    batch = next(iter(runtime.train_batches))

    assert sorted(batch.lengths.tolist()) == [1, 3]
    assert batch.static_fc.shape == (2, 3, 3)
    assert batch.dynamic_fc.shape == (2, 3, 3, 3)

    for row, length in enumerate(batch.lengths.tolist()):
        if length < batch.dynamic_fc.shape[1]:
            assert torch.all(batch.dynamic_fc[row, length:] == 0)


def test_fusion_runtime_requires_both_data_artifacts(tmp_path) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _fusion_candidate(split_path)

    static_h5.unlink()
    with pytest.raises(FileNotFoundError, match="Static ABIDE HDF5"):
        _factory(
            split_path,
            static_h5,
            temporal_h5,
        )(candidate, 0, tmp_path / "run_static_missing")

    _write_static_h5(static_h5)
    temporal_h5.unlink()
    with pytest.raises(FileNotFoundError, match="Temporal ABIDE HDF5"):
        _factory(
            split_path,
            static_h5,
            temporal_h5,
        )(candidate, 0, tmp_path / "run_temporal_missing")


def test_fusion_runtime_provenance_records_both_artifacts_without_test_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _fusion_candidate(split_path)

    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: nn.Linear(1, 1),
    )

    runtime = _factory(
        split_path,
        static_h5,
        temporal_h5,
    )(candidate, 5, tmp_path / "run")

    provenance = dict(runtime.user_provenance)
    assert provenance["mode"] == "fusion"
    assert provenance["data_artifact"] == {
        "static_h5": str(static_h5),
        "temporal_h5": str(temporal_h5),
    }
    assert provenance["train_n"] == 2
    assert provenance["val_n"] == 1
    assert provenance["candidate_index"] == 5
    assert all("test" not in key.lower() for key in provenance)


def test_fusion_runtime_and_factory_model_execute_one_real_batch(tmp_path) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _fusion_candidate(split_path)

    runtime = _factory(
        split_path,
        static_h5,
        temporal_h5,
    )(candidate, 0, tmp_path / "run")
    batch = next(iter(runtime.train_batches))

    runtime.model.eval()
    with torch.no_grad():
        output = runtime.model(
            batch.static_fc,
            batch.dynamic_fc,
            batch.lengths,
        )

    assert output.logits.shape == (2, 2)
    assert torch.isfinite(output.logits).all()
