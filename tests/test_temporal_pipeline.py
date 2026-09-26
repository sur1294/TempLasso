from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig
from unified_bn.temporal.windowing import WindowPolicyConfig


class CountingBackbone(BaseBackbone):
    def __init__(self, n_rois: int = 4, embedding_dim: int = 6) -> None:
        super().__init__(n_rois=n_rois, embedding_dim=embedding_dim)
        self.last_batch_size: int | None = None
        self.linear = nn.Linear(n_rois * n_rois, embedding_dim)

    def _encode_impl(self, fc: torch.Tensor) -> BackboneOutput:
        self.last_batch_size = int(fc.shape[0])
        embedding = self.linear(fc.reshape(fc.shape[0], -1))
        return BackboneOutput(embedding=embedding)


def _make_pipeline(
    *,
    window_policy: WindowPolicyConfig | None = None,
) -> TemporalPipeline:
    backbone = CountingBackbone()
    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=6,
            hidden_dim=5,
            pooling="final",
        )
    )
    return TemporalPipeline(
        backbone,
        aggregator,
        num_classes=2,
        window_policy=window_policy,
    )


def test_temporal_pipeline_only_encodes_valid_windows() -> None:
    pipeline = _make_pipeline()

    x = torch.randn(3, 6, 4, 4)
    lengths = torch.tensor([6, 4, 2], dtype=torch.long)

    out = pipeline(x, lengths, return_aux=True)

    assert pipeline.backbone.last_batch_size == 12
    assert out.aux["n_valid_windows"] == 12
    assert out.aux["window_embeddings"].shape == (3, 6, 6)


def test_temporal_pipeline_padding_values_do_not_change_prediction() -> None:
    torch.manual_seed(9)
    pipeline = _make_pipeline()
    pipeline.eval()

    x1 = torch.randn(3, 6, 4, 4)
    lengths = torch.tensor([6, 4, 2], dtype=torch.long)

    x2 = x1.clone()
    x2[1, 4:] = 1e6 * torch.randn_like(x2[1, 4:])
    x2[2, 2:] = 1e6 * torch.randn_like(x2[2, 2:])

    with torch.no_grad():
        y1 = pipeline(x1, lengths)
        y2 = pipeline(x2, lengths)

    torch.testing.assert_close(
        y1.temporal_embedding,
        y2.temporal_embedding,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        y1.logits,
        y2.logits,
        rtol=0.0,
        atol=0.0,
    )


def test_temporal_pipeline_valid_window_change_can_change_prediction() -> None:
    torch.manual_seed(12)
    pipeline = _make_pipeline()
    pipeline.eval()

    x1 = torch.randn(2, 5, 4, 4)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    x2 = x1.clone()
    x2[0, 1] = x2[0, 1] + 5.0

    with torch.no_grad():
        y1 = pipeline(x1, lengths).logits
        y2 = pipeline(x2, lengths).logits

    assert not torch.equal(y1, y2)


def test_temporal_pipeline_gradients_reach_backbone_gru_classifier() -> None:
    pipeline = _make_pipeline()
    pipeline.train()

    x = torch.randn(2, 5, 4, 4)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    loss = pipeline(x, lengths).logits.square().mean()
    loss.backward()

    assert pipeline.backbone.linear.weight.grad is not None
    assert pipeline.temporal_aggregator.gru.weight_ih_l0.grad is not None
    assert pipeline.classifier.weight.grad is not None


def test_temporal_pipeline_min_policy_uses_exact_common_length() -> None:
    pipeline = _make_pipeline(
        window_policy=WindowPolicyConfig(
            policy="min",
            min_windows=3,
            crop="start",
        )
    )

    x = torch.randn(3, 6, 4, 4)
    lengths = torch.tensor([6, 5, 3], dtype=torch.long)

    out = pipeline(x, lengths, return_aux=True)

    assert pipeline.backbone.last_batch_size == 9
    assert torch.equal(
        out.aux["effective_lengths"],
        torch.tensor([3, 3, 3]),
    )
    assert out.aux["window_embeddings"].shape == (3, 3, 6)


def test_temporal_pipeline_dimension_mismatch_rejected() -> None:
    backbone = CountingBackbone(embedding_dim=6)
    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=7,
            hidden_dim=4,
        )
    )

    with pytest.raises(ValueError, match="must match"):
        TemporalPipeline(backbone, aggregator)
