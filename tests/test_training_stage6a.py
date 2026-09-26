from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.config import (
    CandidateRunConfig,
    EarlyStoppingConfig,
    RunIdentity,
    TrainingConfig,
    ValidationSelectionRule,
)
from unified_bn.training.metrics import (
    BinaryMetricAccumulator,
    BinaryMetricError,
    UndefinedBinaryMetricError,
)
from unified_bn.training.reproducibility import (
    make_dataloader_generator,
    seed_dataloader_worker,
    seed_everything,
)


def test_atlas_abide_candidate_defaults_are_explicit() -> None:
    config = TrainingConfig()

    assert config.epochs == 200
    assert config.batch_size == 16
    assert config.optimizer.name == "adam"
    assert config.optimizer.lr == pytest.approx(1e-4)
    assert config.optimizer.weight_decay == pytest.approx(1e-4)
    assert config.lr_schedule.name == "cosine"
    assert config.lr_schedule.min_lr == pytest.approx(1e-5)

    assert config.checkpoint_selection.metric == "val_loss"
    assert config.checkpoint_selection.mode == "min"
    assert config.early_stopping.enabled is False

    # The contract itself is machine-readable and JSON-safe.
    json.dumps(config.to_dict(), allow_nan=False, sort_keys=True)


def test_validation_selection_is_strict_and_separate_from_early_stopping() -> None:
    rule = ValidationSelectionRule(metric="val_loss", mode="min")

    assert rule.is_better(0.40, 0.50)
    assert not rule.is_better(0.50, 0.50)  # historical strict '<' behavior
    assert not rule.is_better(0.60, 0.50)

    stopping = EarlyStoppingConfig(enabled=False)
    assert stopping.enabled is False
    assert stopping.rule == ValidationSelectionRule()

    with pytest.raises(ValueError, match="positive patience"):
        EarlyStoppingConfig(enabled=True, patience=None)


def test_run_identity_enforces_mode_specific_provenance_and_serializes() -> None:
    vanilla = RunIdentity(
        mode="vanilla",
        seed=42,
        split_identity="sha256:example-static-split",
        backbone_config={
            "name": "bqn",
            "n_layers": 3,
            "activation": "elu",
            "dropout": 0.3,
        },
    )
    assert vanilla.to_dict()["temporal_config"] is None
    assert vanilla.to_dict()["window_policy"] is None

    temporal = RunIdentity(
        mode="temporal",
        seed=42,
        split_identity="sha256:example-temporal-split",
        backbone_config={"name": "bqn", "n_layers": 3},
        temporal_config={
            "name": "bigru",
            "hidden_dim": 32,
            "num_layers": 1,
            "pooling": "final",
        },
        window_policy=WindowPolicyConfig(
            policy="min",
            crop="center",
            min_windows=12,
        ),
    )
    payload = CandidateRunConfig(identity=temporal).to_dict()
    assert payload["identity"]["window_policy"] == {
        "policy": "min",
        "crop": "center",
        "min_windows": 12,
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)

    with pytest.raises(ValueError, match="must not define temporal_config"):
        RunIdentity(
            mode="vanilla",
            seed=42,
            split_identity="sha256:bad",
            backbone_config={"name": "bqn"},
            temporal_config={"name": "bigru"},
        )

    with pytest.raises(ValueError, match="requires window_policy"):
        RunIdentity(
            mode="temporal",
            seed=42,
            split_identity="sha256:bad",
            backbone_config={"name": "bqn"},
            temporal_config={"name": "bigru"},
            window_policy=None,
        )


def test_seed_everything_reproduces_python_numpy_and_torch_cpu() -> None:
    seed_everything(12345, deterministic_algorithms=True, warn_only=False)
    first = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )

    seed_everything(12345, deterministic_algorithms=True, warn_only=False)
    second = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)


def test_dataloader_generator_reproduces_shuffle_order() -> None:
    order1 = torch.randperm(64, generator=make_dataloader_generator(1042))
    order2 = torch.randperm(64, generator=make_dataloader_generator(1042))
    order3 = torch.randperm(64, generator=make_dataloader_generator(2042))

    assert torch.equal(order1, order2)
    assert not torch.equal(order1, order3)


def test_seed_dataloader_worker_reproduces_python_and_numpy_streams() -> None:
    torch.manual_seed(2026)
    seed_dataloader_worker(0)
    first = (random.random(), float(np.random.random()))

    torch.manual_seed(2026)
    seed_dataloader_worker(0)
    second = (random.random(), float(np.random.random()))

    assert first == second


def _tiny_training_run(seed: int) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Small CPU-only determinism regression independent of Stage 6C."""

    seed_everything(seed, deterministic_algorithms=True, warn_only=False)

    features = torch.arange(48, dtype=torch.float32).reshape(24, 2) / 47.0
    labels = ((features[:, 0] + features[:, 1]) > 1.0).to(torch.long)

    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=6,
        shuffle=True,
        generator=make_dataloader_generator(seed),
        num_workers=0,
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses: list[float] = []
    for _ in range(3):
        epoch_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            epoch_sum += float(loss.detach().item()) * y.numel()
        losses.append(epoch_sum / len(features))

    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return state, losses


def test_same_seed_reproduces_tiny_synthetic_training() -> None:
    state1, losses1 = _tiny_training_run(3042)
    state2, losses2 = _tiny_training_run(3042)

    assert losses1 == losses2
    assert state1.keys() == state2.keys()
    for name in state1:
        torch.testing.assert_close(state1[name], state2[name], rtol=0.0, atol=0.0)


def test_binary_metrics_match_known_argmax_and_auc_example() -> None:
    # P(class=1) = [0.2, 0.9, 0.8, 0.7]
    # labels       = [0,   1,   0,   1]
    # argmax pred  = [0,   1,   1,   1]
    # AUC=0.75, ACC=0.75, SEN=1.0, SPE=0.5.
    probabilities = torch.tensor([0.2, 0.9, 0.8, 0.7], dtype=torch.float64)
    logits = torch.stack(
        [torch.log1p(-probabilities), torch.log(probabilities)],
        dim=1,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    accumulator = BinaryMetricAccumulator()
    accumulator.update(
        logits=logits[:2],
        labels=labels[:2],
        loss_sum=F.cross_entropy(logits[:2], labels[:2], reduction="sum"),
    )
    accumulator.update(
        logits=logits[2:],
        labels=labels[2:],
        loss_sum=F.cross_entropy(logits[2:], labels[2:], reduction="sum"),
    )
    metrics = accumulator.compute()

    full_loss = F.cross_entropy(logits, labels, reduction="sum").item() / 4
    assert metrics.loss == pytest.approx(full_loss)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.sensitivity == pytest.approx(1.0)
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.n_samples == 4

    # Compatibility aliases preserve historical BQN naming without changing
    # the canonical Stage 6 field names.
    assert metrics.auc == metrics.roc_auc
    assert metrics.acc == metrics.accuracy
    assert metrics.sens == metrics.sensitivity
    assert metrics.spec == metrics.specificity


def test_binary_metric_accumulation_is_batch_partition_invariant() -> None:
    logits = torch.tensor(
        [[3.0, 0.0], [0.0, 2.0], [0.0, 1.0], [2.0, 0.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    one = BinaryMetricAccumulator()
    one.update(
        logits=logits,
        labels=labels,
        loss_sum=F.cross_entropy(logits, labels, reduction="sum"),
    )

    two = BinaryMetricAccumulator()
    for slc in (slice(0, 1), slice(1, 4)):
        two.update(
            logits=logits[slc],
            labels=labels[slc],
            loss_sum=F.cross_entropy(
                logits[slc], labels[slc], reduction="sum"
            ),
        )

    metrics_one = one.compute()
    metrics_two = two.compute()

    assert metrics_one.loss == pytest.approx(metrics_two.loss, rel=0.0, abs=1e-15)
    assert metrics_one.roc_auc == metrics_two.roc_auc
    assert metrics_one.accuracy == metrics_two.accuracy
    assert metrics_one.sensitivity == metrics_two.sensitivity
    assert metrics_one.specificity == metrics_two.specificity
    assert metrics_one.n_samples == metrics_two.n_samples


def test_zero_predicted_positives_is_valid_when_ground_truth_has_both_classes() -> None:
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    logits = torch.tensor(
        [[3.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )

    accumulator = BinaryMetricAccumulator()
    accumulator.update(
        logits=logits,
        labels=labels,
        loss_sum=F.cross_entropy(logits, labels, reduction="sum"),
    )
    metrics = accumulator.compute()

    assert metrics.sensitivity == pytest.approx(0.0)
    assert metrics.specificity == pytest.approx(1.0)


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
def test_one_class_ground_truth_fails_loudly(labels: list[int]) -> None:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    logits = torch.zeros((len(labels), 2), dtype=torch.float64)

    accumulator = BinaryMetricAccumulator()
    accumulator.update(
        logits=logits,
        labels=labels_tensor,
        loss_sum=F.cross_entropy(logits, labels_tensor, reduction="sum"),
    )

    with pytest.raises(UndefinedBinaryMetricError, match="require both"):
        accumulator.compute()


def test_binary_metric_input_validation_is_fail_closed() -> None:
    accumulator = BinaryMetricAccumulator()

    with pytest.raises(BinaryMetricError, match="shape"):
        accumulator.update(
            logits=torch.zeros(2, 1),  # a [B,K] head needs K >= 2
            labels=torch.tensor([0, 1]),
            loss_sum=1.0,
        )

    with pytest.raises(BinaryMetricError, match="only 0/1"):
        accumulator.update(
            logits=torch.zeros(2, 2),
            labels=torch.tensor([0, 2]),
            loss_sum=1.0,
        )

    with pytest.raises(BinaryMetricError, match="zero accumulated samples"):
        BinaryMetricAccumulator().compute()
