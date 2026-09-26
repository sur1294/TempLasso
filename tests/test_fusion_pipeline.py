from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput
from unified_bn.fusion.concat import ConcatenationFusion
from unified_bn.pipelines.fusion import FusionPipeline
from unified_bn.temporal.bigru import (
    BiGRUAggregator,
    BiGRUConfig,
)
from unified_bn.temporal.windowing import WindowPolicyConfig


class CountingBackbone(BaseBackbone):
    def __init__(
        self,
        *,
        n_rois: int = 4,
        embedding_dim: int = 6,
    ) -> None:
        super().__init__(
            n_rois=n_rois,
            embedding_dim=embedding_dim,
        )
        self.linear = nn.Linear(
            n_rois * n_rois,
            embedding_dim,
        )
        self.seen_batch_sizes: list[int] = []

    def _encode_impl(
        self,
        fc: torch.Tensor,
    ) -> BackboneOutput:
        self.seen_batch_sizes.append(
            int(fc.shape[0])
        )
        embedding = self.linear(
            fc.reshape(fc.shape[0], -1)
        )
        return BackboneOutput(
            embedding=embedding,
        )


def _make_pipeline(
    *,
    window_policy: WindowPolicyConfig | None = None,
) -> FusionPipeline:
    static_backbone = CountingBackbone(
        embedding_dim=6,
    )
    temporal_backbone = CountingBackbone(
        embedding_dim=6,
    )
    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=6,
            hidden_dim=5,
            pooling="final",
        )
    )
    fusion = ConcatenationFusion(
        static_dim=6,
        temporal_dim=10,
    )

    return FusionPipeline(
        static_backbone,
        temporal_backbone,
        aggregator,
        fusion,
        num_classes=2,
        window_policy=window_policy,
    )


def test_fusion_pipeline_declares_expected_dimensions() -> None:
    pipeline = _make_pipeline()

    assert pipeline.n_rois == 4
    assert pipeline.static_backbone.embedding_dim == 6
    assert pipeline.temporal_backbone.embedding_dim == 6
    assert pipeline.temporal_aggregator.output_dim == 10
    assert pipeline.fusion.output_dim == 16
    assert pipeline.fusion_dim == 16

    assert isinstance(
        pipeline.classifier,
        nn.Linear,
    )
    assert pipeline.classifier.in_features == 16
    assert pipeline.classifier.out_features == 2


def test_fusion_pipeline_only_encodes_valid_temporal_windows() -> None:
    pipeline = _make_pipeline()

    static_fc = torch.randn(3, 4, 4)
    dynamic_fc = torch.randn(3, 6, 4, 4)
    lengths = torch.tensor(
        [6, 4, 2],
        dtype=torch.long,
    )

    output = pipeline(
        static_fc,
        dynamic_fc,
        lengths,
        return_aux=True,
    )

    assert pipeline.static_backbone.seen_batch_sizes == [3]
    assert pipeline.temporal_backbone.seen_batch_sizes == [12]

    assert output.aux["n_valid_windows"] == 12
    assert output.aux["window_embeddings"].shape == (
        3,
        6,
        6,
    )


def test_fusion_pipeline_output_shapes_are_correct() -> None:
    pipeline = _make_pipeline()

    output = pipeline(
        torch.randn(3, 4, 4),
        torch.randn(3, 5, 4, 4),
        torch.tensor([5, 3, 2]),
    )

    assert output.static_embedding.shape == (3, 6)
    assert output.temporal_embedding.shape == (3, 10)
    assert output.fused_embedding.shape == (3, 16)
    assert output.logits.shape == (3, 2)
    assert output.aux == {}


def test_fused_embedding_is_exact_static_temporal_concatenation() -> None:
    torch.manual_seed(19)
    pipeline = _make_pipeline()
    pipeline.eval()

    with torch.no_grad():
        output = pipeline(
            torch.randn(2, 4, 4),
            torch.randn(2, 4, 4, 4),
            torch.tensor([4, 2]),
        )

    expected = torch.cat(
        (
            output.static_embedding,
            output.temporal_embedding,
        ),
        dim=-1,
    )

    torch.testing.assert_close(
        output.fused_embedding,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_padded_temporal_values_do_not_change_fusion_prediction() -> None:
    torch.manual_seed(23)
    pipeline = _make_pipeline()
    pipeline.eval()

    static_fc = torch.randn(3, 4, 4)
    dynamic_a = torch.randn(3, 6, 4, 4)
    lengths = torch.tensor([6, 4, 2])

    dynamic_b = dynamic_a.clone()
    dynamic_b[1, 4:] = (
        1e6 * torch.randn_like(dynamic_b[1, 4:])
    )
    dynamic_b[2, 2:] = (
        1e6 * torch.randn_like(dynamic_b[2, 2:])
    )

    with torch.no_grad():
        first = pipeline(
            static_fc,
            dynamic_a,
            lengths,
        )
        second = pipeline(
            static_fc,
            dynamic_b,
            lengths,
        )

    torch.testing.assert_close(
        first.temporal_embedding,
        second.temporal_embedding,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.fused_embedding,
        second.fused_embedding,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.logits,
        second.logits,
        rtol=0.0,
        atol=0.0,
    )


def test_static_input_change_can_change_fusion_prediction() -> None:
    torch.manual_seed(29)
    pipeline = _make_pipeline()
    pipeline.eval()

    static_a = torch.randn(2, 4, 4)
    static_b = static_a.clone()
    static_b[0] += 5.0

    dynamic_fc = torch.randn(2, 4, 4, 4)
    lengths = torch.tensor([4, 3])

    with torch.no_grad():
        first = pipeline(
            static_a,
            dynamic_fc,
            lengths,
        ).logits
        second = pipeline(
            static_b,
            dynamic_fc,
            lengths,
        ).logits

    assert not torch.equal(first, second)


def test_valid_temporal_change_can_change_fusion_prediction() -> None:
    torch.manual_seed(31)
    pipeline = _make_pipeline()
    pipeline.eval()

    static_fc = torch.randn(2, 4, 4)
    dynamic_a = torch.randn(2, 5, 4, 4)
    dynamic_b = dynamic_a.clone()
    dynamic_b[0, 1] += 5.0

    lengths = torch.tensor([5, 3])

    with torch.no_grad():
        first = pipeline(
            static_fc,
            dynamic_a,
            lengths,
        ).logits
        second = pipeline(
            static_fc,
            dynamic_b,
            lengths,
        ).logits

    assert not torch.equal(first, second)


def test_gradients_reach_both_backbones_bigru_and_classifier() -> None:
    pipeline = _make_pipeline()
    pipeline.train()

    output = pipeline(
        torch.randn(2, 4, 4),
        torch.randn(2, 5, 4, 4),
        torch.tensor([5, 3]),
    )

    loss = output.logits.square().mean()
    loss.backward()

    assert (
        pipeline.static_backbone.linear.weight.grad
        is not None
    )
    assert (
        pipeline.temporal_backbone.linear.weight.grad
        is not None
    )
    assert (
        pipeline.temporal_aggregator.gru.weight_ih_l0.grad
        is not None
    )
    assert pipeline.classifier.weight.grad is not None


def test_min_window_policy_is_applied_before_temporal_backbone() -> None:
    pipeline = _make_pipeline(
        window_policy=WindowPolicyConfig(
            policy="min",
            min_windows=3,
            crop="start",
        )
    )

    output = pipeline(
        torch.randn(3, 4, 4),
        torch.randn(3, 6, 4, 4),
        torch.tensor([6, 5, 3]),
        return_aux=True,
    )

    assert pipeline.temporal_backbone.seen_batch_sizes == [9]
    assert torch.equal(
        output.aux["effective_lengths"],
        torch.tensor([3, 3, 3]),
    )
    assert output.aux["window_embeddings"].shape == (
        3,
        3,
        6,
    )


def test_static_temporal_batch_size_mismatch_is_rejected() -> None:
    pipeline = _make_pipeline()

    with pytest.raises(
        ValueError,
        match="identical batch size",
    ):
        pipeline(
            torch.randn(3, 4, 4),
            torch.randn(2, 5, 4, 4),
            torch.tensor([5, 3]),
        )


def test_roi_count_mismatch_between_backbones_is_rejected() -> None:
    static_backbone = CountingBackbone(
        n_rois=4,
        embedding_dim=6,
    )
    temporal_backbone = CountingBackbone(
        n_rois=5,
        embedding_dim=6,
    )
    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=6,
            hidden_dim=5,
        )
    )
    fusion = ConcatenationFusion(
        static_dim=6,
        temporal_dim=10,
    )

    with pytest.raises(
        ValueError,
        match="same ROI count",
    ):
        FusionPipeline(
            static_backbone,
            temporal_backbone,
            aggregator,
            fusion,
        )


def test_temporal_dimension_mismatch_is_rejected() -> None:
    static_backbone = CountingBackbone(
        embedding_dim=6,
    )
    temporal_backbone = CountingBackbone(
        embedding_dim=6,
    )
    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=7,
            hidden_dim=5,
        )
    )
    fusion = ConcatenationFusion(
        static_dim=6,
        temporal_dim=10,
    )

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        FusionPipeline(
            static_backbone,
            temporal_backbone,
            aggregator,
            fusion,
        )


def test_fusion_static_dimension_mismatch_is_rejected() -> None:
    static_backbone = CountingBackbone(
        embedding_dim=6,
    )
    temporal_backbone = CountingBackbone(
        embedding_dim=6,
    )
    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=6,
            hidden_dim=5,
        )
    )
    fusion = ConcatenationFusion(
        static_dim=7,
        temporal_dim=10,
    )

    with pytest.raises(
        ValueError,
        match="fusion static_dim",
    ):
        FusionPipeline(
            static_backbone,
            temporal_backbone,
            aggregator,
            fusion,
        )


def test_fusion_temporal_dimension_mismatch_is_rejected() -> None:
    static_backbone = CountingBackbone(
        embedding_dim=6,
    )
    temporal_backbone = CountingBackbone(
        embedding_dim=6,
    )
    aggregator = BiGRUAggregator(
        BiGRUConfig(
            input_dim=6,
            hidden_dim=5,
        )
    )
    fusion = ConcatenationFusion(
        static_dim=6,
        temporal_dim=11,
    )

    with pytest.raises(
        ValueError,
        match="fusion temporal_dim",
    ):
        FusionPipeline(
            static_backbone,
            temporal_backbone,
            aggregator,
            fusion,
        )
