from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_bn.data.collate import FusionBatch
from unified_bn.training.epoch import run_binary_epoch


class DummyFusionPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 2)
        self.forward_training_flags: list[bool] = []
        self.forward_grad_flags: list[bool] = []
        self.seen_static_fc: list[torch.Tensor] = []
        self.seen_dynamic_fc: list[torch.Tensor] = []
        self.seen_lengths: list[torch.Tensor] = []

    def forward(
        self,
        static_fc: torch.Tensor,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
    ):
        self.forward_training_flags.append(self.training)
        self.forward_grad_flags.append(torch.is_grad_enabled())
        self.seen_static_fc.append(static_fc.detach().clone())
        self.seen_dynamic_fc.append(dynamic_fc.detach().clone())
        self.seen_lengths.append(lengths.detach().clone())

        # Tiny synthetic feature map. The purpose here is to validate the
        # generic epoch-runner fusion dispatch, not fusion mathematics.
        features = torch.stack(
            (
                static_fc[:, 0, 0],
                static_fc[:, 0, 1],
                dynamic_fc[:, 0, 0, 0],
                dynamic_fc[:, 0, 0, 1],
            ),
            dim=1,
        )
        logits = self.classifier(features)
        return SimpleNamespace(logits=logits)


def _fusion_batch() -> FusionBatch:
    static_fc = torch.tensor(
        [
            [[1.0, 0.2], [0.2, 1.0]],
            [[1.0, -0.3], [-0.3, 1.0]],
        ],
        dtype=torch.float32,
    )
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
    return FusionBatch(
        subject_ids=("FUSION_SUBJECT_A", "FUSION_SUBJECT_B"),
        labels=torch.tensor([0, 1], dtype=torch.long),
        static_fc=static_fc,
        dynamic_fc=dynamic_fc,
        lengths=torch.tensor([2, 3], dtype=torch.long),
        static_source_indices=torch.tensor([101, 202], dtype=torch.long),
        temporal_source_indices=torch.tensor([303, 404], dtype=torch.long),
    )


def test_fusion_dispatch_passes_three_predictive_inputs_exactly():
    model = DummyFusionPipeline()
    batch = _fusion_batch()

    metrics = run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device="cpu",
        training=False,
    )

    assert metrics.n_samples == 2
    assert len(model.seen_static_fc) == 1
    assert len(model.seen_dynamic_fc) == 1
    assert len(model.seen_lengths) == 1

    torch.testing.assert_close(
        model.seen_static_fc[0],
        batch.static_fc,
    )
    torch.testing.assert_close(
        model.seen_dynamic_fc[0],
        batch.dynamic_fc,
    )
    torch.testing.assert_close(
        model.seen_lengths[0],
        batch.lengths,
    )


def test_fusion_metadata_is_not_forwarded_to_model():
    # DummyFusionPipeline.forward accepts exactly three predictive tensors.
    # If subject IDs or either source-index field were forwarded, Python would
    # raise immediately.
    model = DummyFusionPipeline()
    batch = _fusion_batch()

    run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device="cpu",
        training=False,
    )

    assert len(model.seen_static_fc) == 1


def test_fusion_training_updates_parameters_and_uses_train_mode():
    torch.manual_seed(41)
    model = DummyFusionPipeline()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    batch = _fusion_batch()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    metrics = run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device="cpu",
        training=True,
        optimizer=optimizer,
    )

    assert metrics.n_samples == 2
    assert model.training is True
    assert model.forward_training_flags == [True]
    assert model.forward_grad_flags == [True]

    assert any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter in model.named_parameters()
    )


def test_fusion_evaluation_uses_eval_mode_and_disables_gradients():
    model = DummyFusionPipeline()
    batch = _fusion_batch()

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


def test_fusion_batch_to_contract_receives_device_and_non_blocking(monkeypatch):
    model = DummyFusionPipeline()
    batch = _fusion_batch()

    calls: list[tuple[torch.device, bool]] = []
    original_to = FusionBatch.to

    def recording_to(
        self,
        device,
        *,
        non_blocking=False,
    ):
        calls.append(
            (
                torch.device(device),
                bool(non_blocking),
            )
        )
        return original_to(
            self,
            device,
            non_blocking=non_blocking,
        )

    monkeypatch.setattr(
        FusionBatch,
        "to",
        recording_to,
    )

    run_binary_epoch(
        model,
        [batch],
        loss_fn=F.cross_entropy,
        device=torch.device("cpu"),
        training=False,
        non_blocking=True,
    )

    assert calls == [
        (torch.device("cpu"), True)
    ]
