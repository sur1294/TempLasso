from __future__ import annotations

import io

import pytest
import torch

from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig


def test_bigru_final_pooling_shape() -> None:
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=32,
            hidden_dim=16,
            num_layers=2,
            dropout=0.2,
            pooling="final",
        )
    )
    x = torch.randn(4, 7, 32)
    lengths = torch.tensor([7, 5, 3, 1], dtype=torch.long)

    out = model(x, lengths)

    assert out.embedding.shape == (4, 32)
    assert out.aux["sequence_output"].shape == (4, 7, 32)
    assert out.aux["final_hidden"].shape == (4, 32)


@pytest.mark.parametrize("pooling", ["final", "mean", "attention"])
def test_bigru_padding_invariance(pooling: str) -> None:
    torch.manual_seed(5)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=8,
            hidden_dim=6,
            pooling=pooling,
            attention_hidden_dim=5,
        )
    )
    model.eval()

    x = torch.randn(3, 6, 8)
    lengths = torch.tensor([6, 4, 2], dtype=torch.long)

    x_changed = x.clone()
    x_changed[1, 4:] = 1e4 * torch.randn_like(x_changed[1, 4:])
    x_changed[2, 2:] = 1e4 * torch.randn_like(x_changed[2, 2:])

    with torch.no_grad():
        y1 = model(x, lengths).embedding
        y2 = model(x_changed, lengths).embedding

    torch.testing.assert_close(y1, y2, rtol=0.0, atol=0.0)


def test_bigru_attention_weights_are_masked_and_normalized() -> None:
    torch.manual_seed(8)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=10,
            hidden_dim=7,
            pooling="attention",
            attention_hidden_dim=6,
        )
    )
    model.eval()

    x = torch.randn(2, 5, 10)
    lengths = torch.tensor([5, 2], dtype=torch.long)

    with torch.no_grad():
        out = model(x, lengths)

    weights = out.aux["attention_weights"]

    assert weights.shape == (2, 5)
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(2),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(weights[1, 2:], torch.zeros(3))


def test_bigru_unsorted_lengths_supported() -> None:
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=6,
            hidden_dim=4,
            pooling="final",
        )
    )
    x = torch.randn(4, 5, 6)
    lengths = torch.tensor([2, 5, 1, 4], dtype=torch.long)

    out = model(x, lengths)
    assert out.embedding.shape == (4, 8)


def test_bigru_single_window_subject_supported() -> None:
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=3,
            pooling="mean",
        )
    )
    x = torch.randn(2, 4, 5)
    lengths = torch.tensor([1, 4], dtype=torch.long)

    out = model(x, lengths)
    assert out.embedding.shape == (2, 6)


def test_bigru_projection_and_layernorm_shape() -> None:
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=12,
            hidden_dim=9,
            pooling="attention",
            projection_dim=11,
            layer_norm=True,
        )
    )
    x = torch.randn(3, 4, 12)
    lengths = torch.tensor([4, 3, 2], dtype=torch.long)

    out = model(x, lengths)

    assert model.output_dim == 11
    assert out.embedding.shape == (3, 11)


def test_bigru_gradients_reach_gru_and_attention() -> None:
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=8,
            hidden_dim=5,
            pooling="attention",
            attention_hidden_dim=4,
        )
    )

    x = torch.randn(3, 5, 8, requires_grad=True)
    lengths = torch.tensor([5, 4, 2], dtype=torch.long)

    loss = model(x, lengths).embedding.square().mean()
    loss.backward()

    assert x.grad is not None
    assert model.gru.weight_ih_l0.grad is not None
    assert model.pooler.scorer[0].weight.grad is not None
    assert model.pooler.scorer[-1].weight.grad is not None


def test_bigru_state_dict_roundtrip() -> None:
    torch.manual_seed(33)
    cfg = BiGRUConfig(
        input_dim=8,
        hidden_dim=5,
        num_layers=2,
        dropout=0.1,
        pooling="final",
        projection_dim=7,
        layer_norm=True,
    )

    model1 = BiGRUAggregator(cfg)
    model1.eval()

    buffer = io.BytesIO()
    torch.save(model1.state_dict(), buffer)
    buffer.seek(0)

    model2 = BiGRUAggregator(cfg)
    model2.load_state_dict(torch.load(buffer, map_location="cpu"))
    model2.eval()

    x = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    with torch.no_grad():
        y1 = model1(x, lengths).embedding
        y2 = model2(x, lengths).embedding

    torch.testing.assert_close(y1, y2, rtol=0.0, atol=0.0)
