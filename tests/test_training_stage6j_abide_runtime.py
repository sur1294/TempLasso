from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

import unified_bn.training.abide_runtime as runtime_module
from unified_bn.data import StaticBatch, TemporalBatch, load_frozen_split
from unified_bn.training.abide_runtime import (
    ABIDERuntimeConfig,
    ABIDERuntimeError,
    ABIDERuntimeFactory,
    frozen_split_identity,
    read_stored_split_seed,
)
from unified_bn.training.search_space import (
    BQNSearchSpace,
    BiGRUSearchSpace,
    generate_temporal_bqn_bigru_candidates,
    generate_vanilla_bqn_candidates,
)
from unified_bn.temporal.windowing import WindowPolicyConfig


SUBJECTS = ("s1", "s2", "s3", "s4")
LABELS = np.asarray([0, 1, 0, 1], dtype=np.int64)


def _write_split(path: Path, *, seed: int = 42) -> None:
    np.savez(
        path,
        train_ids=np.asarray(["s2", "s1"]),
        val_ids=np.asarray(["s3"]),
        test_ids=np.asarray(["s4"]),
        seed=np.asarray(seed, dtype=np.int64),
    )


def _write_static_h5(path: Path) -> None:
    fc = np.zeros((4, 3, 3), dtype=np.float32)
    for i in range(4):
        fc[i] = np.eye(3, dtype=np.float32)
        fc[i, 0, 1] = fc[i, 1, 0] = 0.1 * (i + 1)

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "subject_ids",
            data=np.asarray(SUBJECTS, dtype="S"),
        )
        h5.create_dataset("labels", data=LABELS)
        h5.create_dataset("fc", data=fc)


def _write_temporal_h5(path: Path) -> None:
    lengths = np.asarray([1, 2, 1, 2], dtype=np.int64)
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(lengths)]
    )
    rows, cols = np.triu_indices(3, k=1)

    windows = int(lengths.sum())
    edges = np.zeros((windows, len(rows)), dtype=np.float32)
    for i in range(windows):
        edges[i] = 0.01 * (i + 1)

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "subject_ids",
            data=np.asarray(SUBJECTS, dtype="S"),
        )
        h5.create_dataset("labels", data=LABELS)
        h5.create_dataset("K", data=lengths)
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


def _identity(split_path: Path) -> str:
    return frozen_split_identity(load_frozen_split(split_path))


def _vanilla_candidate(split_path: Path):
    return generate_vanilla_bqn_candidates(
        seed=42,
        split_identity=_identity(split_path),
        backbone_space=BQNSearchSpace(
            n_layers=(1,),
            activations=("elu",),
            dropouts=(0.0,),
            n_rois=3,
            cluster_num=2,
        ),
    )[0]


def _temporal_candidate(split_path: Path):
    return generate_temporal_bqn_bigru_candidates(
        seed=42,
        split_identity=_identity(split_path),
        window_policy=WindowPolicyConfig(policy="native"),
        backbone_space=BQNSearchSpace(
            n_layers=(1,),
            activations=("elu",),
            dropouts=(0.0,),
            n_rois=3,
            cluster_num=2,
        ),
        temporal_space=BiGRUSearchSpace(
            hidden_dims=(4,),
            num_layers=(1,),
            poolings=("final",),
            stacked_dropouts=(0.0,),
        ),
    )[0]


def _factory_config(
    split_path: Path,
    static_h5: Path,
    temporal_h5: Path,
):
    return ABIDERuntimeConfig(
        split_path=split_path,
        static_h5=static_h5,
        temporal_h5=temporal_h5,
        device="cpu",
        expected_n_rois=3,
        pin_memory=False,
    )


def test_frozen_split_identity_is_path_independent_and_order_sensitive(
    tmp_path,
) -> None:
    first_path = tmp_path / "a.npz"
    second_path = tmp_path / "b.npz"
    _write_split(first_path)
    _write_split(second_path)

    first = load_frozen_split(first_path)
    second = load_frozen_split(second_path)

    assert frozen_split_identity(first) == frozen_split_identity(second)
    assert frozen_split_identity(first).startswith("sha256:")
    assert len(frozen_split_identity(first)) == len("sha256:") + 64

    reordered_path = tmp_path / "reordered.npz"
    np.savez(
        reordered_path,
        train_ids=np.asarray(["s1", "s2"]),
        val_ids=np.asarray(["s3"]),
        test_ids=np.asarray(["s4"]),
        seed=np.asarray(42, dtype=np.int64),
    )
    reordered = load_frozen_split(reordered_path)

    assert frozen_split_identity(first) != frozen_split_identity(reordered)


def test_read_stored_split_seed_reads_only_scalar_seed(tmp_path) -> None:
    split_path = tmp_path / "split.npz"
    _write_split(split_path, seed=1042)

    assert read_stored_split_seed(split_path) == 1042


def test_runtime_factory_rejects_candidate_seed_mismatch(tmp_path) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _vanilla_candidate(split_path)
    candidate = replace(
        candidate,
        identity=replace(candidate.identity, seed=1042),
    )

    factory = ABIDERuntimeFactory(
        _factory_config(split_path, static_h5, temporal_h5)
    )

    with pytest.raises(ABIDERuntimeError, match="seed does not match"):
        factory(candidate, 0, tmp_path / "run")


def test_runtime_factory_rejects_split_identity_mismatch(tmp_path) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _vanilla_candidate(split_path)
    candidate = replace(
        candidate,
        identity=replace(
            candidate.identity,
            split_identity="sha256:" + "0" * 64,
        ),
    )

    factory = ABIDERuntimeFactory(
        _factory_config(split_path, static_h5, temporal_h5)
    )

    with pytest.raises(
        ABIDERuntimeError,
        match="split_identity does not match",
    ):
        factory(candidate, 0, tmp_path / "run")


def test_runtime_factory_rejects_roi_mismatch(tmp_path) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _vanilla_candidate(split_path)

    factory = ABIDERuntimeFactory(
        ABIDERuntimeConfig(
            split_path=split_path,
            static_h5=static_h5,
            temporal_h5=temporal_h5,
            device="cpu",
            expected_n_rois=4,
            pin_memory=False,
        )
    )

    with pytest.raises(
        ABIDERuntimeError,
        match="ROI dimension",
    ):
        factory(candidate, 0, tmp_path / "run")


def test_vanilla_runtime_builds_only_frozen_train_and_val(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _vanilla_candidate(split_path)

    sentinel_model = nn.Linear(1, 1)
    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: sentinel_model,
    )

    factory = ABIDERuntimeFactory(
        _factory_config(split_path, static_h5, temporal_h5)
    )
    runtime = factory(candidate, 7, tmp_path / "run")

    assert runtime.model is sentinel_model
    assert tuple(runtime.train_batches.dataset.subject_ids) == ("s2", "s1")
    assert tuple(runtime.val_batches.dataset.subject_ids) == ("s3",)

    assert runtime.train_batches.batch_size == candidate.training.batch_size
    assert runtime.val_batches.batch_size == candidate.training.batch_size
    assert runtime.dataloader_generator is not None

    batch = next(iter(runtime.train_batches))
    assert isinstance(batch, StaticBatch)
    assert batch.fc.shape[1:] == (3, 3)

    # No test loader/dataset is part of the CandidateRuntime contract.
    assert not hasattr(runtime, "test_batches")


def test_temporal_runtime_preserves_native_lengths_and_batch_padding(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _temporal_candidate(split_path)

    sentinel_model = nn.Linear(1, 1)
    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: sentinel_model,
    )

    factory = ABIDERuntimeFactory(
        _factory_config(split_path, static_h5, temporal_h5)
    )
    runtime = factory(candidate, 0, tmp_path / "run")

    assert tuple(runtime.train_batches.dataset.subject_ids) == ("s2", "s1")
    assert runtime.train_batches.dataset.lengths.tolist() == [2, 1]

    batch = next(iter(runtime.train_batches))
    assert isinstance(batch, TemporalBatch)
    assert sorted(batch.lengths.tolist()) == [1, 2]
    assert batch.dynamic_fc.shape == (2, 2, 3, 3)

    for row, length in enumerate(batch.lengths.tolist()):
        if length < batch.dynamic_fc.shape[1]:
            assert torch.all(batch.dynamic_fc[row, length:] == 0)


def test_runtime_uses_candidate_batch_size_and_num_workers(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _vanilla_candidate(split_path)
    candidate = replace(
        candidate,
        training=replace(
            candidate.training,
            batch_size=1,
            num_workers=0,
        ),
    )

    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: nn.Linear(1, 1),
    )

    factory = ABIDERuntimeFactory(
        _factory_config(split_path, static_h5, temporal_h5)
    )
    runtime = factory(candidate, 0, tmp_path / "run")

    assert runtime.train_batches.batch_size == 1
    assert runtime.val_batches.batch_size == 1
    assert runtime.train_batches.num_workers == 0


def test_runtime_provenance_records_artifacts_without_test_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)
    candidate = _vanilla_candidate(split_path)

    monkeypatch.setattr(
        runtime_module,
        "build_model_from_candidate",
        lambda config: nn.Linear(1, 1),
    )

    factory = ABIDERuntimeFactory(
        _factory_config(split_path, static_h5, temporal_h5)
    )
    runtime = factory(candidate, 3, tmp_path / "run")

    provenance = dict(runtime.user_provenance)
    assert provenance["dataset"] == "ABIDE"
    assert provenance["split_identity"] == _identity(split_path)
    assert provenance["split_seed"] == 42
    assert provenance["train_n"] == 2
    assert provenance["val_n"] == 1
    assert provenance["candidate_index"] == 3

    assert all("test" not in key.lower() for key in provenance)


def test_runtime_requires_stored_split_seed_by_default(tmp_path) -> None:
    split_path = tmp_path / "split.npz"
    np.savez(
        split_path,
        train_ids=np.asarray(["s1", "s2"]),
        val_ids=np.asarray(["s3"]),
        test_ids=np.asarray(["s4"]),
    )

    static_h5 = tmp_path / "static.h5"
    temporal_h5 = tmp_path / "temporal.h5"
    _write_static_h5(static_h5)
    _write_temporal_h5(temporal_h5)

    with pytest.raises(ABIDERuntimeError, match="scalar 'seed'"):
        ABIDERuntimeFactory(
            _factory_config(split_path, static_h5, temporal_h5)
        )


def test_pin_memory_defaults_from_device(tmp_path) -> None:
    split_path, static_h5, temporal_h5 = _artifacts(tmp_path)

    cpu_factory = ABIDERuntimeFactory(
        ABIDERuntimeConfig(
            split_path=split_path,
            static_h5=static_h5,
            temporal_h5=temporal_h5,
            device="cpu",
            expected_n_rois=3,
            pin_memory=None,
        )
    )
    assert cpu_factory._resolved_pin_memory() is False


def test_factory_call_has_no_test_data_argument() -> None:
    import inspect

    signature = inspect.signature(ABIDERuntimeFactory.__call__)

    assert tuple(signature.parameters) == (
        "self",
        "candidate",
        "index",
        "run_dir",
    )
    assert all(
        "test" not in name.lower()
        for name in signature.parameters
    )
