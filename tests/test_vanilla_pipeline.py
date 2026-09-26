from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.bqn.bqn_model import BQNConfig, StaticBQN
from unified_bn.backbones.bqn import BQNBackbone, BQNBackboneConfig
from unified_bn.pipelines.vanilla import VanillaPipeline


def _make_reference_bqn(*, seed: int = 1234) -> StaticBQN:
    torch.manual_seed(seed)
    return StaticBQN(
        BQNConfig(
            n_rois=100,
            n_layers=3,
            dropout=0.0,
            activation="leaky_relu",
            cluster_num=4,
            preserve_unused_mlp_b=True,
            orthogonal_centers=True,
            freeze_centers=True,
            project_assignment=True,
        )
    )


def _make_adapter(reference: StaticBQN) -> BQNBackbone:
    return BQNBackbone(
        BQNBackboneConfig(
            n_rois=100,
            n_layers=3,
            dropout=0.0,
            activation="leaky_relu",
            cluster_num=4,
            preserve_unused_mlp_b=True,
            orthogonal_centers=True,
            freeze_centers=True,
            project_assignment=True,
        ),
        model=reference,
    )


def test_vanilla_bqn_embedding_matches_reference_exactly() -> None:
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)
    pipeline = VanillaPipeline(adapter)

    reference.eval()
    pipeline.eval()

    x = torch.randn(4, 100, 100)

    with torch.no_grad():
        expected = reference(x, return_embedding=True)
        observed = pipeline(x).embedding

    assert torch.equal(observed, expected)


def test_vanilla_can_reuse_exact_reference_bqn_classifier() -> None:
    """
    Regression test proving the generic vanilla pipeline can reproduce the
    complete validated StaticBQN prediction path when supplied with the exact
    reference classifier head.
    """
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)

    pipeline = VanillaPipeline(
        adapter,
        num_classes=2,
        classifier=reference.classifier,
    )

    reference.eval()
    pipeline.eval()

    x = torch.randn(5, 100, 100)

    with torch.no_grad():
        expected_logits = reference(x)
        observed = pipeline(x)

    assert torch.equal(observed.logits, expected_logits)


def test_vanilla_reference_classifier_parameters_are_shared() -> None:
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)

    pipeline = VanillaPipeline(
        adapter,
        classifier=reference.classifier,
    )

    reference_classifier_params = dict(reference.classifier.named_parameters())
    pipeline_classifier_params = dict(pipeline.classifier.named_parameters())

    assert reference_classifier_params.keys() == pipeline_classifier_params.keys()

    for name in reference_classifier_params:
        assert (
            reference_classifier_params[name]
            is pipeline_classifier_params[name]
        )


def test_vanilla_default_classifier_shape() -> None:
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)
    pipeline = VanillaPipeline(adapter, num_classes=2)

    x = torch.randn(3, 100, 100)
    out = pipeline(x)

    assert out.embedding.shape == (3, 32)
    assert out.logits.shape == (3, 2)
    assert out.aux == {}


def test_vanilla_aux_passthrough_is_explicit() -> None:
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)
    pipeline = VanillaPipeline(adapter)

    x = torch.randn(2, 100, 100)

    without_aux = pipeline(x, return_aux=False)
    with_aux = pipeline(x, return_aux=True)

    assert without_aux.aux == {}
    assert "assignment" in with_aux.aux
    assert "topo" in with_aux.aux
    assert with_aux.aux["assignment"].shape == (2, 100, 4)
    assert with_aux.aux["topo"].shape == (2, 100, 100)


def test_vanilla_gradients_reach_backbone_and_classifier() -> None:
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)
    pipeline = VanillaPipeline(adapter)

    pipeline.train()

    x = torch.randn(2, 100, 100)
    out = pipeline(x)
    loss = out.logits.square().mean()
    loss.backward()

    assert pipeline.classifier.weight.grad is not None

    for index, layer in enumerate(reference.qp_layers):
        assert layer.mlp_a.weight.grad is not None, (
            f"layer {index} mlp_a did not receive gradient"
        )
        assert layer.mlp_h.weight.grad is not None, (
            f"layer {index} mlp_h did not receive gradient"
        )

        if layer.mlp_b is not None:
            assert layer.mlp_b.weight.grad is None


class WrongShapeClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_vanilla_rejects_classifier_output_shape_violation() -> None:
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)

    pipeline = VanillaPipeline(
        adapter,
        num_classes=2,
        classifier=WrongShapeClassifier(),
    )

    with pytest.raises(ValueError, match="output contract"):
        pipeline(torch.randn(2, 100, 100))


def test_vanilla_rejects_invalid_num_classes() -> None:
    reference = _make_reference_bqn()
    adapter = _make_adapter(reference)

    with pytest.raises(ValueError, match="at least 2"):
        VanillaPipeline(adapter, num_classes=1)
