from __future__ import annotations

import copy

import pytest
import torch

from models.bqn.bqn_model import BQNConfig, StaticBQN
from unified_bn.backbones.bqn import BQNBackbone, BQNBackboneConfig


def _make_reference_and_adapter(
    *,
    seed: int = 1234,
    dropout: float = 0.0,
):
    """
    Construct one validated StaticBQN and wrap that exact object.

    Using the same object is the strongest possible regression check for the
    adapter path because any numerical difference must come from the adapter,
    not from independently randomized cluster centers or weights.
    """
    torch.manual_seed(seed)

    ref_cfg = BQNConfig(
        n_rois=100,
        n_layers=3,
        dropout=dropout,
        activation="leaky_relu",
        cluster_num=4,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )
    reference = StaticBQN(ref_cfg)

    adapter_cfg = BQNBackboneConfig(
        n_rois=100,
        n_layers=3,
        dropout=dropout,
        activation="leaky_relu",
        cluster_num=4,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )
    adapter = BQNBackbone(adapter_cfg, model=reference)

    return reference, adapter


def test_bqn_adapter_embedding_is_exact_reference_embedding() -> None:
    reference, adapter = _make_reference_and_adapter(dropout=0.0)

    reference.eval()
    adapter.eval()

    x = torch.randn(4, 100, 100)

    with torch.no_grad():
        expected = reference(x, return_embedding=True)
        observed = adapter.encode(x)

    assert expected.shape == (4, 32)
    assert observed.shape == (4, 32)

    # Same underlying model and same execution path should be bitwise equal.
    assert torch.equal(observed, expected)


def test_bqn_adapter_aux_is_exact_reference_aux() -> None:
    reference, adapter = _make_reference_and_adapter(dropout=0.0)

    reference.eval()
    adapter.eval()

    x = torch.randn(3, 100, 100)

    with torch.no_grad():
        expected_embedding, expected_aux = reference(
            x,
            return_embedding=True,
            return_aux=True,
        )
        observed = adapter(x)

    assert torch.equal(observed.embedding, expected_embedding)
    assert torch.equal(observed.aux["assignment"], expected_aux["assignment"])
    assert torch.equal(observed.aux["topo"], expected_aux["topo"])


def test_bqn_adapter_classify_is_exact_reference_logits() -> None:
    reference, adapter = _make_reference_and_adapter(dropout=0.0)

    reference.eval()
    adapter.eval()

    x = torch.randn(5, 100, 100)

    with torch.no_grad():
        expected = reference(x)
        observed = adapter.classify(x)

    assert expected.shape == (5, 2)
    assert torch.equal(observed, expected)


def test_bqn_adapter_preserves_parameter_identity_when_wrapping_model() -> None:
    reference, adapter = _make_reference_and_adapter(dropout=0.0)

    reference_params = dict(reference.named_parameters())
    adapter_params = dict(adapter.reference_model.named_parameters())

    assert reference_params.keys() == adapter_params.keys()

    for name in reference_params:
        assert reference_params[name] is adapter_params[name]


def test_bqn_adapter_gradients_reach_reference_bqn() -> None:
    reference, adapter = _make_reference_and_adapter(dropout=0.0)

    reference.train()
    adapter.train()

    x = torch.randn(2, 100, 100)

    embedding = adapter.encode(x)
    loss = embedding.square().mean()
    loss.backward()

    for index, layer in enumerate(reference.qp_layers):
        assert layer.mlp_a.weight.grad is not None, (
            f"layer {index} mlp_a did not receive gradient"
        )
        assert layer.mlp_h.weight.grad is not None, (
            f"layer {index} mlp_h did not receive gradient"
        )

        if layer.mlp_b is not None:
            assert layer.mlp_b.weight.grad is None, (
                "Reference-inactive mlp_b unexpectedly received gradient."
            )


def test_bqn_adapter_declares_reference_embedding_dimension() -> None:
    reference, adapter = _make_reference_and_adapter(dropout=0.0)

    assert reference.embedding_dim == 32
    assert adapter.embedding_dim == 32
    assert adapter.n_rois == 100


def test_bqn_adapter_rejects_configuration_mismatch() -> None:
    torch.manual_seed(1)

    reference = StaticBQN(
        BQNConfig(
            n_rois=100,
            n_layers=4,
            dropout=0.0,
            activation="elu",
            cluster_num=4,
        )
    )

    incompatible_adapter_config = BQNBackboneConfig(
        n_rois=100,
        n_layers=3,
        dropout=0.0,
        activation="leaky_relu",
        cluster_num=4,
    )

    with pytest.raises(ValueError, match="does not match"):
        BQNBackbone(
            incompatible_adapter_config,
            model=reference,
        )


def test_bqn_adapter_can_construct_reference_model_from_config() -> None:
    torch.manual_seed(10)

    cfg = BQNBackboneConfig(
        n_rois=100,
        n_layers=5,
        dropout=0.3,
        activation="gelu",
        cluster_num=4,
    )

    adapter = BQNBackbone(cfg)

    assert isinstance(adapter.reference_model, StaticBQN)
    assert adapter.reference_model.config.n_layers == 5
    assert adapter.reference_model.config.dropout == 0.3
    assert adapter.reference_model.config.activation == "gelu"
    assert adapter.embedding_dim == 32
