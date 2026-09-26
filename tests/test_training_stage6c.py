from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_bn.data.collate import StaticBatch, TemporalBatch
from unified_bn.training.epoch import EpochRunnerError, run_binary_epoch


class DummyStaticPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 2)
        self.forward_training_flags: list[bool] = []
        self.forward_grad_flags: list[bool] = []
        self.seen_fc: list[torch.Tensor] = []

    def forward(self, fc: torch.Tensor):
        self.forward_training_flags.append(self.training)
        self.forward_grad_flags.append(torch.is_grad_enabled())
        self.seen_fc.append(fc.detach().clone())
        logits = self.classifier(fc.reshape(fc.shape[0], -1))
        return SimpleNamespace(logits=logits)


class DummyTemporalPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)
        self.forward_training_flags: list[bool] = []
        self.forward_grad_flags: list[bool] = []
        self.seen_dynamic_fc: list[torch.Tensor] = []
        self.seen_lengths: list[torch.Tensor] = []

    def forward(self, dynamic_fc: torch.Tensor, lengths: torch.Tensor):
        self.forward_training_flags.append(self.training)
        self.forward_grad_flags.append(torch.is_grad_enabled())
        self.seen_dynamic_fc.append(dynamic_fc.detach().clone())
        self.seen_lengths.append(lengths.detach().clone())

        # Use the first two scalar entries only to keep this a tiny synthetic
        # pipeline. The temporal dispatch contract, not temporal modeling, is
        # what Stage 6C tests here.
        features = dynamic_fc[:, 0, 0, :2]
        logits = self.classifier(features)
        return SimpleNamespace(logits=logits)


class FixedLogitPipeline(nn.Module):
    def forward(self, fc: torch.Tensor):
        # fc is [B,1,2] and directly carries predetermined binary logits.
        return SimpleNamespace(logits=fc[:, 0, :])


def _static_batch(
    values: torch.Tensor,
    labels: list[int],
    *,
    prefix: str = "s",
) -> StaticBatch:
    batch_size = values.shape[0]
    return StaticBatch(
        subject_ids=tuple(f"{prefix}{i}" for i in range(batch_size)),
        labels=torch.tensor(labels, dtype=torch.long),
        fc=values.to(dtype=torch.float32),
        source_indices=torch.arange(batch_size, dtype=torch.long),
    )


def _temporal_batch() -> TemporalBatch:
    dynamic_fc = torch.tensor(
        [
            [
                [[1.0, 0.1], [0.1, 1.0]],
                [[1.0, 0.2], [0.2, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[1.0, -0.1], [-0.1, 1.0]],
                [[1.0, -0.2], [-0.2, 1.0]],
                [[1.0, -0.3], [-0.3, 1.0]],
            ],
        ],
        dtype=torch.float32,
    )
    return TemporalBatch(
        subject_ids=("101", "202"),
        labels=torch.tensor([0, 1], dtype=torch.long),
        dynamic_fc=dynamic_fc,
        lengths=torch.tensor([2, 3], dtype=torch.long),
        source_indices=torch.tensor([11, 22], dtype=torch.long),
    )


def test_static_training_dispatch_updates_parameters_and_uses_train_mode():
    torch.manual_seed(7)
    model = DummyStaticPipeline()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    batches = [
        _static_batch(
            torch.tensor(
                [
                    [[1.0, 0.2], [0.2, 1.0]],
                    [[1.0, -0.3], [-0.3, 1.0]],
                ]
            ),
            [0, 1],
        ),
        _static_batch(
            torch.tensor(
                [
                    [[1.0, 0.4], [0.4, 1.0]],
                    [[1.0, -0.5], [-0.5, 1.0]],
                ]
            ),
            [0, 1],
            prefix="t",
        ),
    ]

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    metrics = run_binary_epoch(
        model,
        batches,
        loss_fn=F.cross_entropy,
        device="cpu",
        training=True,
        optimizer=optimizer,
    )

    assert metrics.n_samples == 4
    assert model.training is True
    assert model.forward_training_flags == [True, True]
    assert model.forward_grad_flags == [True, True]
    assert len(model.seen_fc) == 2

    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_temporal_dispatch_passes_dynamic_fc_and_lengths_exactly():
    torch.manual_seed(9)
    model = DummyTemporalPipeline()
    batch = _temporal_batch()

    metrics = run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device="cpu",
        training=False,
    )

    assert metrics.n_samples == 2
    assert len(model.seen_dynamic_fc) == 1
    assert len(model.seen_lengths) == 1
    torch.testing.assert_close(model.seen_dynamic_fc[0], batch.dynamic_fc)
    torch.testing.assert_close(model.seen_lengths[0], batch.lengths)


def test_subject_ids_and_source_indices_are_metadata_not_model_inputs():
    # The dummy forward signature accepts only fc. If metadata were forwarded,
    # Python would raise immediately. Distinctive metadata makes the intent
    # explicit rather than relying on placeholder values.
    model = DummyStaticPipeline()
    batch = StaticBatch(
        subject_ids=("DO_NOT_FORWARD_A", "DO_NOT_FORWARD_B"),
        labels=torch.tensor([0, 1], dtype=torch.long),
        fc=torch.eye(2).repeat(2, 1, 1),
        source_indices=torch.tensor([987654, 123456], dtype=torch.long),
    )

    run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device="cpu",
        training=False,
    )

    assert len(model.seen_fc) == 1


def test_evaluation_sets_eval_mode_and_disables_gradients():
    model = DummyStaticPipeline()
    batch = _static_batch(
        torch.eye(2).repeat(2, 1, 1),
        [0, 1],
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device="cpu",
        training=False,
    )

    assert model.training is False
    assert model.forward_training_flags == [False]
    assert model.forward_grad_flags == [False]

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name])
        assert parameter.grad is None


def test_training_requires_optimizer():
    model = DummyStaticPipeline()
    batch = _static_batch(torch.eye(2).repeat(2, 1, 1), [0, 1])

    with pytest.raises(EpochRunnerError, match="requires an optimizer"):
        run_binary_epoch(
            model,
            [batch],
            loss_fn=F.cross_entropy,
            device="cpu",
            training=True,
            optimizer=None,
        )


def test_evaluation_rejects_optimizer_to_prevent_accidental_steps():
    model = DummyStaticPipeline()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = _static_batch(torch.eye(2).repeat(2, 1, 1), [0, 1])

    with pytest.raises(EpochRunnerError, match="must not receive an optimizer"):
        run_binary_epoch(
            model,
            [batch],
            loss_fn=F.cross_entropy,
            device="cpu",
            training=False,
            optimizer=optimizer,
        )


def test_unsupported_batch_type_fails_loudly():
    model = DummyStaticPipeline()

    with pytest.raises(TypeError, match="Unsupported batch type"):
        run_binary_epoch(
            model,
            [(torch.zeros(2, 2, 2), torch.tensor([0, 1]))],
            loss_fn=F.cross_entropy,
            device="cpu",
            training=False,
        )


def test_pipeline_output_must_expose_logits():
    class BadPipeline(nn.Module):
        def forward(self, fc: torch.Tensor):
            return SimpleNamespace(embedding=fc.mean(dim=(1, 2)))

    batch = _static_batch(torch.eye(2).repeat(2, 1, 1), [0, 1])

    with pytest.raises(EpochRunnerError, match="'logits'"):
        run_binary_epoch(
            BadPipeline(),
            [batch],
            loss_fn=F.cross_entropy,
            device="cpu",
            training=False,
        )


def test_loss_function_must_return_scalar_mean_loss():
    model = DummyStaticPipeline()
    batch = _static_batch(torch.eye(2).repeat(2, 1, 1), [0, 1])

    def unreduced_loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels, reduction="none")

    with pytest.raises(EpochRunnerError, match="scalar mean per-sample loss"):
        run_binary_epoch(
            model,
            [batch],
            loss_fn=unreduced_loss,
            device="cpu",
            training=False,
        )


def test_nonfinite_loss_fails_loudly():
    model = DummyStaticPipeline()
    batch = _static_batch(torch.eye(2).repeat(2, 1, 1), [0, 1])

    def nan_loss(logits: torch.Tensor, labels: torch.Tensor):
        return logits.sum() * torch.tensor(float("nan"))

    with pytest.raises(EpochRunnerError, match="NaN or Inf"):
        run_binary_epoch(
            model,
            [batch],
            loss_fn=nan_loss,
            device="cpu",
            training=False,
        )


def test_epoch_loss_is_sample_weighted_across_unequal_batches():
    model = FixedLogitPipeline()

    batch1 = _static_batch(
        torch.tensor(
            [
                [[2.0, 0.0]],
                [[0.0, 2.0]],
                [[1.5, 0.5]],
            ]
        ),
        [0, 1, 0],
    )
    batch2 = _static_batch(
        torch.tensor([[[0.2, 1.8]]]),
        [1],
        prefix="u",
    )

    metrics = run_binary_epoch(
        model,
        [batch1, batch2],
        loss_fn=F.cross_entropy,
        device="cpu",
        training=False,
    )

    all_logits = torch.cat([batch1.fc[:, 0, :], batch2.fc[:, 0, :]], dim=0)
    all_labels = torch.cat([batch1.labels, batch2.labels], dim=0)
    expected_loss = F.cross_entropy(all_logits, all_labels, reduction="mean")

    assert metrics.n_samples == 4
    assert metrics.loss == pytest.approx(float(expected_loss.item()), abs=1e-7)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.sensitivity == pytest.approx(1.0)
    assert metrics.specificity == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)


def test_batch_to_contract_is_used_with_requested_device_and_non_blocking(monkeypatch):
    model = DummyStaticPipeline()
    batch = _static_batch(torch.eye(2).repeat(2, 1, 1), [0, 1])

    calls: list[tuple[torch.device, bool]] = []
    original_to = StaticBatch.to

    def recording_to(self, device, *, non_blocking=False):
        calls.append((torch.device(device), bool(non_blocking)))
        return original_to(self, device, non_blocking=non_blocking)

    monkeypatch.setattr(StaticBatch, "to", recording_to)

    run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device=torch.device("cpu"),
        training=False,
        non_blocking=True,
    )

    assert calls == [(torch.device("cpu"), True)]


def test_empty_epoch_fails_instead_of_returning_fabricated_metrics():
    model = DummyStaticPipeline()

    with pytest.raises(ValueError, match="zero accumulated samples"):
        run_binary_epoch(
            model,
            [],
            loss_fn=F.cross_entropy,
            device="cpu",
            training=False,
        )
