from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from unified_bn.data.splits import FrozenSplit
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.checkpoint import (
    CHECKPOINT_SCHEMA_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointFormatError,
    CheckpointProgress,
    capture_rng_state,
    frozen_split_identity,
    load_training_checkpoint,
    read_training_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from unified_bn.training.config import (
    CandidateRunConfig,
    OptimizerConfig,
    RunIdentity,
    TrainingConfig,
)
from unified_bn.training.reproducibility import (
    make_dataloader_generator,
    seed_everything,
)


def _vanilla_config(
    *,
    seed: int = 42,
    split_identity: str = "sha256:split-A",
    layers: int = 3,
    lr: float = 1e-4,
) -> CandidateRunConfig:
    return CandidateRunConfig(
        identity=RunIdentity(
            mode="vanilla",
            seed=seed,
            split_identity=split_identity,
            backbone_config={
                "name": "bqn",
                "layers": layers,
                "activation": "elu",
                "dropout": 0.3,
            },
        ),
        training=TrainingConfig(
            optimizer=OptimizerConfig(lr=lr, weight_decay=1e-4)
        ),
    )


def _temporal_config(
    *,
    seed: int = 42,
    split_identity: str = "sha256:split-A",
    hidden_dim: int = 32,
    window_policy: WindowPolicyConfig | None = None,
) -> CandidateRunConfig:
    return CandidateRunConfig(
        identity=RunIdentity(
            mode="temporal",
            seed=seed,
            split_identity=split_identity,
            backbone_config={
                "name": "bqn",
                "layers": 3,
                "activation": "elu",
                "dropout": 0.3,
            },
            temporal_config={
                "name": "bigru",
                "hidden_dim": hidden_dim,
                "num_layers": 1,
                "pooling": "final",
            },
            window_policy=(
                WindowPolicyConfig(policy="native")
                if window_policy is None
                else window_policy
            ),
        )
    )


def _training_objects():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=5,
        eta_min=1e-5,
    )
    return model, optimizer, scheduler


def _one_step(model, optimizer, scheduler, x, y) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach().item())


def _assert_state_dict_equal(first, second) -> None:
    assert first.keys() == second.keys()
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        else:
            assert left == right


def test_frozen_split_identity_is_semantic_order_sensitive_and_stable() -> None:
    split = FrozenSplit(
        train_ids=("50030", "50002"),
        val_ids=("51585",),
        test_ids=("60001",),
    )
    same = FrozenSplit(
        train_ids=("50030", "50002"),
        val_ids=("51585",),
        test_ids=("60001",),
    )
    reordered = FrozenSplit(
        train_ids=("50002", "50030"),
        val_ids=("51585",),
        test_ids=("60001",),
    )

    identity = frozen_split_identity(split)
    assert identity.startswith("sha256:")
    assert identity == frozen_split_identity(same)
    assert identity != frozen_split_identity(reordered)


def test_checkpoint_progress_is_one_based_and_fail_closed() -> None:
    progress = CheckpointProgress(
        completed_epoch=7,
        best_epoch=4,
        best_selection_value=0.512,
    )
    assert progress.to_dict() == {
        "completed_epoch": 7,
        "best_epoch": 4,
        "best_selection_value": pytest.approx(0.512),
    }

    with pytest.raises(ValueError, match="completed_epoch"):
        CheckpointProgress(0, 0, 0.5)
    with pytest.raises(ValueError, match="best_epoch"):
        CheckpointProgress(3, 4, 0.5)
    with pytest.raises(ValueError, match="finite"):
        CheckpointProgress(3, 2, float("nan"))


def test_checkpoint_roundtrip_restores_model_optimizer_scheduler_and_progress(
    tmp_path: Path,
) -> None:
    seed_everything(2026, deterministic_algorithms=True, warn_only=False)
    model, optimizer, scheduler = _training_objects()
    x = torch.randn(8, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0])
    _one_step(model, optimizer, scheduler, x, y)

    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())

    config = _vanilla_config()
    progress = CheckpointProgress(1, 1, 0.61)
    path = tmp_path / "candidate.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_config=config,
        progress=progress,
        user_provenance={"git_commit": "test-commit"},
    )

    restored_model, restored_optimizer, restored_scheduler = _training_objects()
    restored_progress = load_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_run_config=config,
        restore_rng=False,
    )

    assert restored_progress == progress
    _assert_state_dict_equal(expected_model, restored_model.state_dict())

    # Optimizer state has nested tensors; compare the serialized state exactly.
    restored_optimizer_state = restored_optimizer.state_dict()
    assert expected_optimizer["param_groups"] == restored_optimizer_state["param_groups"]
    assert expected_optimizer["state"].keys() == restored_optimizer_state["state"].keys()
    for parameter_id in expected_optimizer["state"]:
        for field, expected_value in expected_optimizer["state"][parameter_id].items():
            actual_value = restored_optimizer_state["state"][parameter_id][field]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(expected_value, actual_value, rtol=0.0, atol=0.0)
            else:
                assert expected_value == actual_value

    assert restored_scheduler.state_dict() == expected_scheduler

    loaded = read_training_checkpoint(path, expected_run_config=config)
    assert loaded.user_provenance == {"git_commit": "test-commit"}
    assert loaded.runtime_provenance["torch_version"] == str(torch.__version__)


def test_checkpoint_file_uses_explicit_versioned_safe_schema(tmp_path: Path) -> None:
    model, optimizer, scheduler = _training_objects()
    config = _vanilla_config()
    path = tmp_path / "schema.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_config=config,
        progress=CheckpointProgress(1, 1, 0.9),
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_name"] == CHECKPOINT_SCHEMA_NAME
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert "test" not in payload
    assert "test_metrics" not in payload
    assert payload["run_config"] == config.to_dict()


def test_read_validates_compatibility_before_any_live_state_is_mutated(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _training_objects()
    path = tmp_path / "candidate.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_config=_vanilla_config(seed=42),
        progress=CheckpointProgress(1, 1, 0.7),
    )

    fresh_model, _, _ = _training_objects()
    before = copy.deepcopy(fresh_model.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="identity.seed"):
        read_training_checkpoint(
            path,
            expected_run_config=_vanilla_config(seed=1042),
        )

    _assert_state_dict_equal(before, fresh_model.state_dict())


@pytest.mark.parametrize(
    "checkpoint_config, expected_config, expected_fragment",
    [
        (_vanilla_config(seed=42), _vanilla_config(seed=1042), "identity.seed"),
        (
            _vanilla_config(split_identity="sha256:A"),
            _vanilla_config(split_identity="sha256:B"),
            "identity.split_identity",
        ),
        (
            _vanilla_config(layers=3),
            _vanilla_config(layers=5),
            "identity.backbone_config.layers",
        ),
        (
            _vanilla_config(lr=1e-4),
            _vanilla_config(lr=2e-4),
            "training.optimizer.lr",
        ),
        (
            _temporal_config(hidden_dim=32),
            _temporal_config(hidden_dim=64),
            "identity.temporal_config.hidden_dim",
        ),
        (
            _temporal_config(window_policy=WindowPolicyConfig(policy="native")),
            _temporal_config(
                window_policy=WindowPolicyConfig(
                    policy="min", crop="center", min_windows=12
                )
            ),
            "identity.window_policy",
        ),
        (
            _vanilla_config(),
            _temporal_config(),
            "identity.mode",
        ),
    ],
)
def test_incompatible_candidate_resume_is_rejected(
    tmp_path: Path,
    checkpoint_config: CandidateRunConfig,
    expected_config: CandidateRunConfig,
    expected_fragment: str,
) -> None:
    model, optimizer, scheduler = _training_objects()
    path = tmp_path / "candidate.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_config=checkpoint_config,
        progress=CheckpointProgress(1, 1, 0.8),
    )

    with pytest.raises(CheckpointCompatibilityError, match=expected_fragment):
        read_training_checkpoint(path, expected_run_config=expected_config)


def test_scheduler_presence_and_object_type_mismatch_fail_before_restore(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _training_objects()
    config = _vanilla_config()
    path = tmp_path / "candidate.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_config=config,
        progress=CheckpointProgress(1, 1, 0.8),
    )
    loaded = read_training_checkpoint(path, expected_run_config=config)

    new_model, new_optimizer, _ = _training_objects()
    with pytest.raises(CheckpointCompatibilityError, match="scheduler"):
        restore_training_checkpoint(
            loaded,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=None,
            restore_rng=False,
        )

    wrong_optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-3)
    _, _, matching_scheduler = _training_objects()
    with pytest.raises(CheckpointCompatibilityError, match="optimizer"):
        restore_training_checkpoint(
            loaded,
            model=new_model,
            optimizer=wrong_optimizer,
            scheduler=matching_scheduler,
            restore_rng=False,
        )


def test_rng_roundtrip_restores_python_numpy_torch_and_dataloader_generator() -> None:
    seed_everything(4042, deterministic_algorithms=True, warn_only=False)
    generator = make_dataloader_generator(4042)

    # Advance all streams before capturing them.
    random.random()
    np.random.random()
    torch.rand(3)
    torch.randperm(10, generator=generator)

    snapshot = capture_rng_state(dataloader_generator=generator)

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        torch.randperm(12, generator=generator),
    )

    # Disturb all streams, then restore to the checkpoint boundary.
    for _ in range(5):
        random.random()
        np.random.random()
        torch.rand(4)
        torch.randperm(12, generator=generator)

    restore_rng_state(snapshot, dataloader_generator=generator)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        torch.randperm(12, generator=generator),
    )

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    torch.testing.assert_close(expected[2], actual[2], rtol=0.0, atol=0.0)
    assert torch.equal(expected[3], actual[3])


def test_rng_restore_requires_dataloader_generator_when_checkpoint_contains_one() -> None:
    generator = make_dataloader_generator(42)
    snapshot = capture_rng_state(dataloader_generator=generator)

    with pytest.raises(CheckpointCompatibilityError, match="DataLoader generator"):
        restore_rng_state(snapshot, dataloader_generator=None)


def test_exact_epoch_boundary_resume_reproduces_uninterrupted_training(
    tmp_path: Path,
) -> None:
    seed = 3042
    config = _vanilla_config(seed=seed)

    seed_everything(seed, deterministic_algorithms=True, warn_only=False)
    uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler = _training_objects()
    uninterrupted_generator = make_dataloader_generator(seed)

    x = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 35.0
    y = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])

    order_epoch1 = torch.randperm(len(x), generator=uninterrupted_generator)
    loss1 = _one_step(
        uninterrupted_model,
        uninterrupted_optimizer,
        uninterrupted_scheduler,
        x[order_epoch1],
        y[order_epoch1],
    )

    checkpoint_path = tmp_path / "resume.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=uninterrupted_model,
        optimizer=uninterrupted_optimizer,
        scheduler=uninterrupted_scheduler,
        run_config=config,
        progress=CheckpointProgress(1, 1, loss1),
        dataloader_generator=uninterrupted_generator,
    )

    # Uninterrupted epoch 2.
    order_epoch2 = torch.randperm(len(x), generator=uninterrupted_generator)
    uninterrupted_loss2 = _one_step(
        uninterrupted_model,
        uninterrupted_optimizer,
        uninterrupted_scheduler,
        x[order_epoch2],
        y[order_epoch2],
    )
    uninterrupted_state = copy.deepcopy(uninterrupted_model.state_dict())
    uninterrupted_optimizer_state = copy.deepcopy(uninterrupted_optimizer.state_dict())
    uninterrupted_scheduler_state = copy.deepcopy(uninterrupted_scheduler.state_dict())

    # Fresh process-equivalent objects consume RNG during construction, then
    # checkpoint restoration resets all saved streams to the epoch boundary.
    resumed_model, resumed_optimizer, resumed_scheduler = _training_objects()
    resumed_generator = make_dataloader_generator(seed + 999)
    progress = load_training_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_run_config=config,
        restore_rng=True,
        dataloader_generator=resumed_generator,
    )
    assert progress.completed_epoch == 1

    resumed_order_epoch2 = torch.randperm(len(x), generator=resumed_generator)
    assert torch.equal(order_epoch2, resumed_order_epoch2)

    resumed_loss2 = _one_step(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        x[resumed_order_epoch2],
        y[resumed_order_epoch2],
    )

    assert uninterrupted_loss2 == resumed_loss2
    _assert_state_dict_equal(uninterrupted_state, resumed_model.state_dict())

    resumed_optimizer_state = resumed_optimizer.state_dict()
    assert uninterrupted_optimizer_state["param_groups"] == resumed_optimizer_state["param_groups"]
    for parameter_id in uninterrupted_optimizer_state["state"]:
        for field, expected_value in uninterrupted_optimizer_state["state"][parameter_id].items():
            actual_value = resumed_optimizer_state["state"][parameter_id][field]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(expected_value, actual_value, rtol=0.0, atol=0.0)
            else:
                assert expected_value == actual_value

    assert resumed_scheduler.state_dict() == uninterrupted_scheduler_state


def test_corrupt_or_wrong_schema_checkpoint_is_rejected(tmp_path: Path) -> None:
    config = _vanilla_config()

    wrong_schema = tmp_path / "wrong-schema.pt"
    torch.save(
        {
            "schema_name": "other.schema",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_config": config.to_dict(),
            "progress": {},
            "model_state": {},
            "optimizer_state": {},
            "scheduler_state": None,
            "rng_state": {},
            "object_types": {},
            "runtime_provenance": {},
            "user_provenance": {},
        },
        wrong_schema,
    )
    with pytest.raises(CheckpointFormatError, match="Unsupported checkpoint schema"):
        read_training_checkpoint(wrong_schema, expected_run_config=config)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointFormatError, match="Could not safely read"):
        read_training_checkpoint(corrupt, expected_run_config=config)
