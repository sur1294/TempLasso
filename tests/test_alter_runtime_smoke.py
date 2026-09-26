"""
End-to-end validation search for a vanilla ALTER candidate on synthetic
ABIDE-shaped artifacts.

This is the test that proves ALTER is wired into the *existing* machinery --
frozen split, runtime factory, trainer, checkpointing, validation-only
selection -- rather than into a parallel code path of its own. It also pins
the boundary that matters most scientifically: the held-out test subjects are
never read during a search.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import unified_bn.data.abide_static as abide_static_module
from unified_bn.data import load_frozen_split
from unified_bn.training.abide_runtime import (
    ABIDERuntimeConfig,
    ABIDERuntimeFactory,
    frozen_split_identity,
)
from unified_bn.training.checkpoint import read_training_checkpoint
from unified_bn.training.config import (
    LRScheduleConfig,
    OptimizerConfig,
    TrainingConfig,
)
from unified_bn.training.model_factory import build_model_from_candidate
from unified_bn.training.orchestrator import run_validation_search
from unified_bn.training.search_space import (
    ALTERSearchSpace,
    generate_vanilla_alter_candidates,
)


R = 8
SUBJECTS = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")
LABELS = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
TRAIN_IDS = ("s2", "s1", "s5", "s6")
VAL_IDS = ("s3", "s4")
TEST_IDS = ("s7", "s8")

# d_model = n_rois + pos_embed_dim = 16, divisible by nhead=4.
PE_DIM = 8
NHEAD = 4
FFN_DIM = 16
CLUSTER_NUM = 2


def _write_split(path: Path, *, seed: int = 42) -> None:
    np.savez(
        path,
        train_ids=np.asarray(TRAIN_IDS),
        val_ids=np.asarray(VAL_IDS),
        test_ids=np.asarray(TEST_IDS),
        seed=np.asarray(seed, dtype=np.int64),
    )


def _write_static_h5(path: Path) -> None:
    rng = np.random.default_rng(11)
    fc = np.zeros((len(SUBJECTS), R, R), dtype=np.float32)
    for i in range(len(SUBJECTS)):
        noise = rng.normal(0.0, 0.3, size=(R, R)).astype(np.float32)
        matrix = np.tanh(0.5 * (noise + noise.T))
        # Mild class signal on one edge so validation loss can actually move.
        signal = 0.5 if LABELS[i] == 1 else -0.5
        matrix[0, 1] = matrix[1, 0] = signal
        np.fill_diagonal(matrix, 1.0)
        fc[i] = matrix

    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(SUBJECTS, dtype="S"))
        h5.create_dataset("labels", data=LABELS)
        h5.create_dataset("fc", data=fc)


def _write_temporal_h5(path: Path) -> None:
    # Never read by a vanilla candidate, but the runtime config requires the
    # path to exist so the same factory serves every mode.
    lengths = np.asarray([2] * len(SUBJECTS), dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    rows, cols = np.triu_indices(R, k=1)
    windows = int(lengths.sum())

    with h5py.File(path, "w") as h5:
        h5.create_dataset("subject_ids", data=np.asarray(SUBJECTS, dtype="S"))
        h5.create_dataset("labels", data=LABELS)
        h5.create_dataset("K", data=lengths)
        h5.create_dataset("offsets", data=offsets)
        h5.create_dataset(
            "fc_z_upper",
            data=np.zeros((windows, len(rows)), dtype=np.float32),
        )
        h5.create_dataset("triu_row", data=rows.astype(np.int64))
        h5.create_dataset("triu_col", data=cols.astype(np.int64))


@pytest.fixture()
def artifacts(tmp_path: Path):
    split_path = tmp_path / "split.npz"
    static_h5 = tmp_path / "static.h5"
    temporal_h5 = tmp_path / "temporal.h5"
    _write_split(split_path)
    _write_static_h5(static_h5)
    _write_temporal_h5(temporal_h5)
    return split_path, static_h5, temporal_h5


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=2,
        batch_size=2,
        num_workers=0,
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=1e-4),
        lr_schedule=LRScheduleConfig(name="cosine", min_lr=1e-5),
    )


def _candidates(split_path: Path, *, n_layers=(1,)):
    return generate_vanilla_alter_candidates(
        seed=42,
        split_identity=frozen_split_identity(load_frozen_split(split_path)),
        backbone_space=ALTERSearchSpace(
            n_layers=n_layers,
            n_rois=R,
            pos_embed_dim=PE_DIM,
            nhead=NHEAD,
            ffn_dim=FFN_DIM,
            cluster_num=CLUSTER_NUM,
        ),
        training_configs=(_training_config(),),
    )


def _runtime_factory(split_path, static_h5, temporal_h5):
    return ABIDERuntimeFactory(
        ABIDERuntimeConfig(
            split_path=split_path,
            static_h5=static_h5,
            temporal_h5=temporal_h5,
            device="cpu",
            expected_n_rois=R,
            pin_memory=False,
        )
    )


def test_validation_search_runs_end_to_end(artifacts, tmp_path) -> None:
    split_path, static_h5, temporal_h5 = artifacts
    search_dir = tmp_path / "search"
    candidates = _candidates(split_path)

    selection = run_validation_search(
        candidates,
        search_dir=search_dir,
        runtime_factory=_runtime_factory(split_path, static_h5, temporal_h5),
    )

    assert selection.selected_index == 0
    assert selection.selection_metric == "val_loss"
    assert selection.selection_mode == "min"
    assert np.isfinite(selection.selection_value)

    selection_json = search_dir / "selection.json"
    assert selection_json.is_file()
    payload = json.loads(selection_json.read_text())
    assert payload["selected_fingerprint"] == selection.selected_fingerprint

    # The checkpoint has to be loadable *against this candidate*: the reader
    # revalidates the run identity, so this also proves the ALTER candidate
    # round-trips through the resume contract unchanged.
    checkpoint_path = Path(selection.selected_best_checkpoint_path)
    assert checkpoint_path.is_file()
    checkpoint = read_training_checkpoint(
        checkpoint_path, expected_run_config=candidates[0]
    )
    assert checkpoint is not None

    # And the stored weights really are an ALTER state dict.
    model = build_model_from_candidate(candidates[0])
    missing, unexpected = model.load_state_dict(
        checkpoint.model_state, strict=True
    )
    assert not missing and not unexpected


def test_search_selects_between_two_alter_candidates(artifacts, tmp_path) -> None:
    split_path, static_h5, temporal_h5 = artifacts
    candidates = _candidates(split_path, n_layers=(1, 2))
    assert len(candidates) == 2

    selection = run_validation_search(
        candidates,
        search_dir=tmp_path / "search",
        runtime_factory=_runtime_factory(split_path, static_h5, temporal_h5),
    )
    assert selection.selected_index in (0, 1)


def test_provenance_records_the_static_artifact_and_split(
    artifacts, tmp_path
) -> None:
    split_path, static_h5, temporal_h5 = artifacts
    factory = _runtime_factory(split_path, static_h5, temporal_h5)
    candidate = _candidates(split_path)[0]

    runtime = factory(candidate, 0, tmp_path / "run")

    assert runtime.user_provenance["mode"] == "vanilla"
    assert runtime.user_provenance["data_artifact"] == str(static_h5)
    assert runtime.user_provenance["train_n"] == len(TRAIN_IDS)
    assert runtime.user_provenance["val_n"] == len(VAL_IDS)
    # No edge selector is fitted for a non-TempLasso candidate.
    assert "edge_selector" not in runtime.user_provenance


def test_held_out_subjects_are_never_read_during_a_search(
    artifacts, tmp_path, monkeypatch
) -> None:
    split_path, static_h5, temporal_h5 = artifacts
    requested: list[tuple[str, ...]] = []

    original = abide_static_module.StaticABIDEDataset

    class RecordingStaticDataset(original):
        def __init__(self, *args, subject_ids, **kwargs):
            requested.append(tuple(subject_ids))
            super().__init__(*args, subject_ids=subject_ids, **kwargs)

    monkeypatch.setattr(
        "unified_bn.training.abide_runtime.StaticABIDEDataset",
        RecordingStaticDataset,
    )

    run_validation_search(
        _candidates(split_path),
        search_dir=tmp_path / "search",
        runtime_factory=_runtime_factory(split_path, static_h5, temporal_h5),
    )

    assert requested, "expected the static dataset to be constructed"
    assert set(requested) == {TRAIN_IDS, VAL_IDS}
    for subject_ids in requested:
        assert not set(subject_ids) & set(TEST_IDS)


def test_runtime_rejects_an_roi_mismatch(artifacts, tmp_path) -> None:
    split_path, static_h5, temporal_h5 = artifacts
    factory = ABIDERuntimeFactory(
        ABIDERuntimeConfig(
            split_path=split_path,
            static_h5=static_h5,
            temporal_h5=temporal_h5,
            device="cpu",
            expected_n_rois=R + 1,
            pin_memory=False,
        )
    )
    candidate = _candidates(split_path)[0]
    with pytest.raises(Exception):
        factory(candidate, 0, tmp_path / "run")


def test_search_is_reproducible_for_one_seed(artifacts, tmp_path) -> None:
    split_path, static_h5, temporal_h5 = artifacts

    values = []
    for run in ("a", "b"):
        selection = run_validation_search(
            _candidates(split_path),
            search_dir=tmp_path / f"search_{run}",
            runtime_factory=_runtime_factory(
                split_path, static_h5, temporal_h5
            ),
        )
        values.append(selection.selection_value)

    assert values[0] == values[1]


def test_provenance_carries_the_configured_dataset_name(artifacts, tmp_path) -> None:
    # The runtime is dataset-agnostic; the label in provenance must be the one
    # the caller configured (ADNI runs must not be recorded as ABIDE).
    split_path, static_h5, temporal_h5 = artifacts
    factory = ABIDERuntimeFactory(
        ABIDERuntimeConfig(
            split_path=split_path, static_h5=static_h5, temporal_h5=temporal_h5,
            device="cpu", expected_n_rois=R, pin_memory=False, dataset_name="ADNI",
        )
    )
    runtime = factory(_candidates(split_path)[0], 0, tmp_path / "run")
    assert runtime.user_provenance["dataset"] == "ADNI"
    default = _runtime_factory(split_path, static_h5, temporal_h5)(
        _candidates(split_path)[0], 0, tmp_path / "run2"
    )
    assert default.user_provenance["dataset"] == "ABIDE"
