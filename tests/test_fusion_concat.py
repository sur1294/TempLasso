from __future__ import annotations

import pytest
import torch

from unified_bn.fusion.concat import ConcatenationFusion


def test_concat_fusion_declares_exact_dimensions() -> None:
    fusion = ConcatenationFusion(
        static_dim=32,
        temporal_dim=64,
    )

    assert fusion.static_dim == 32
    assert fusion.temporal_dim == 64
    assert fusion.output_dim == 96


def test_concat_fusion_is_exact_tensor_concatenation() -> None:
    h_static = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    h_temporal = torch.tensor(
        [
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0],
        ],
        dtype=torch.float32,
    )

    fusion = ConcatenationFusion(
        static_dim=2,
        temporal_dim=3,
    )

    output = fusion(h_static, h_temporal)

    expected = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 8.0, 9.0, 10.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        output.embedding,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert output.aux == {}


def test_concat_fusion_has_no_trainable_parameters() -> None:
    fusion = ConcatenationFusion(
        static_dim=32,
        temporal_dim=64,
    )

    assert sum(
        parameter.numel()
        for parameter in fusion.parameters()
    ) == 0


def test_concat_fusion_preserves_gradients_to_both_branches() -> None:
    h_static = torch.randn(
        4,
        3,
        requires_grad=True,
    )
    h_temporal = torch.randn(
        4,
        5,
        requires_grad=True,
    )

    fusion = ConcatenationFusion(
        static_dim=3,
        temporal_dim=5,
    )

    loss = fusion(
        h_static,
        h_temporal,
    ).embedding.square().sum()
    loss.backward()

    assert h_static.grad is not None
    assert h_temporal.grad is not None
    assert torch.count_nonzero(h_static.grad) > 0
    assert torch.count_nonzero(h_temporal.grad) > 0


def test_concat_fusion_rejects_batch_size_mismatch() -> None:
    fusion = ConcatenationFusion(
        static_dim=3,
        temporal_dim=5,
    )

    with pytest.raises(
        ValueError,
        match="identical batch size",
    ):
        fusion(
            torch.randn(4, 3),
            torch.randn(3, 5),
        )


def test_concat_fusion_rejects_static_dimension_mismatch() -> None:
    fusion = ConcatenationFusion(
        static_dim=3,
        temporal_dim=5,
    )

    with pytest.raises(
        ValueError,
        match="expected D=3",
    ):
        fusion(
            torch.randn(4, 2),
            torch.randn(4, 5),
        )


def test_concat_fusion_rejects_temporal_dimension_mismatch() -> None:
    fusion = ConcatenationFusion(
        static_dim=3,
        temporal_dim=5,
    )

    with pytest.raises(
        ValueError,
        match="expected D=5",
    ):
        fusion(
            torch.randn(4, 3),
            torch.randn(4, 4),
        )


def test_concat_fusion_rejects_nonfinite_values() -> None:
    fusion = ConcatenationFusion(
        static_dim=2,
        temporal_dim=2,
    )

    h_static = torch.randn(2, 2)
    h_temporal = torch.randn(2, 2)
    h_temporal[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        fusion(
            h_static,
            h_temporal,
        )
