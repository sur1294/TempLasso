"""
``ALTERBackbone`` adapter contract.

The adapter must add nothing: every number it produces has to come out of the
vendored ``StaticALTER``, so that the ALTER baseline stays attributable to the
released implementation rather than to this repository's wrapper.
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from models.alter.alter_model import ALTERConfig, StaticALTER
from unified_bn.backbones import ALTERBackbone, ALTERBackboneConfig
from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput


SMALL = ALTERBackboneConfig(
    n_rois=20, pos_embed_dim=4, nhead=4, ffn_dim=16, cluster_num=3, n_layers=1
)


def _correlation_like(batch_size: int, n_rois: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, n_rois, n_rois, generator=generator)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)
    diagonal = torch.arange(n_rois)
    x[:, diagonal, diagonal] = 1.0
    return x


def test_backbone_config_mirrors_the_reference_config_field_for_field() -> None:
    # _require_exact_keys() in the model factory keys off these field names, and
    # ALTERBackboneConfig.__post_init__ forwards them straight into ALTERConfig,
    # so any drift between the two dataclasses is a silent TypeError waiting to
    # happen.
    # ``num_classes`` is the one reference-only field: the head width is a
    # property of the run (RunIdentity.num_classes), not of the backbone.
    assert {f.name for f in fields(ALTERBackboneConfig)} | {"num_classes"} == {
        f.name for f in fields(ALTERConfig)
    }
    for field in fields(ALTERBackboneConfig):
        reference = next(
            f for f in fields(ALTERConfig) if f.name == field.name
        )
        assert field.default == reference.default, field.name


def test_defaults_build_the_released_abide_model() -> None:
    torch.manual_seed(0)
    backbone = ALTERBackbone()

    assert isinstance(backbone, BaseBackbone)
    assert backbone.n_rois == 100
    assert backbone.embedding_dim == 800
    assert isinstance(backbone.reference_model, StaticALTER)
    assert backbone.reference_model.config == ALTERConfig()


def test_encode_returns_the_reference_pre_classifier_embedding() -> None:
    torch.manual_seed(1)
    backbone = ALTERBackbone(SMALL).eval()
    fc = _correlation_like(4, SMALL.n_rois, 3)

    with torch.no_grad():
        output = backbone(fc)
        expected = backbone.reference_model(fc, return_embedding=True)
        # Must stay inside no_grad: torch only takes the fused attention fast
        # path when grad is disabled, and the fused and math paths differ by
        # ~1e-6 in float32. See
        # test_attention_kernel_differs_between_grad_modes below.
        encoded = backbone.encode(fc)

    assert isinstance(output, BackboneOutput)
    assert output.embedding.shape == (4, 8 * SMALL.cluster_num)
    assert torch.equal(output.embedding, expected)
    assert torch.equal(encoded, expected)


def test_attention_kernel_differs_between_grad_modes() -> None:
    # Not a defect and not something this port introduces: with grad disabled
    # and in eval mode, torch runs the fused encoder-layer kernel; otherwise it
    # runs the `_sa_block` math path that ALTER's `need_weights=True` override
    # sits on. Pinned so nobody later "fixes" a 1e-6 mismatch by changing the
    # model.
    torch.manual_seed(1)
    backbone = ALTERBackbone(SMALL).eval()
    fc = _correlation_like(4, SMALL.n_rois, 3)

    with torch.no_grad():
        fused = backbone.encode(fc)
    math_path = backbone.encode(fc).detach()

    assert not torch.equal(fused, math_path)
    torch.testing.assert_close(fused, math_path, rtol=0.0, atol=1e-5)


def test_aux_exposes_the_reference_diagnostics() -> None:
    torch.manual_seed(2)
    backbone = ALTERBackbone(SMALL).eval()
    fc = _correlation_like(3, SMALL.n_rois, 5)

    with torch.no_grad():
        aux = backbone(fc).aux

    assert set(aux) == {"assignment", "pos_enc", "node_feature"}
    assert aux["assignment"].shape == (3, SMALL.n_rois, SMALL.cluster_num)
    assert aux["pos_enc"].shape == (3, SMALL.n_rois, SMALL.pos_embed_dim)


def test_classify_is_bitwise_the_reference_forward() -> None:
    torch.manual_seed(3)
    backbone = ALTERBackbone(SMALL).eval()
    fc = _correlation_like(4, SMALL.n_rois, 7)

    with torch.no_grad():
        assert torch.equal(
            backbone.classify(fc), backbone.reference_model(fc)
        )


def test_adapter_registers_no_parameters_of_its_own() -> None:
    torch.manual_seed(4)
    backbone = ALTERBackbone(SMALL)

    adapter_ids = {id(p) for p in backbone.parameters()}
    reference_ids = {id(p) for p in backbone.reference_model.parameters()}
    assert adapter_ids == reference_ids

    assert set(backbone.reference_state_dict()) == set(
        backbone.reference_model.state_dict()
    )


def test_adapter_accepts_a_matching_prebuilt_reference_model() -> None:
    torch.manual_seed(5)
    model = StaticALTER(SMALL.to_reference_config())
    backbone = ALTERBackbone(SMALL, model=model)
    assert backbone.reference_model is model


def test_adapter_rejects_a_mismatched_prebuilt_reference_model() -> None:
    torch.manual_seed(6)
    model = StaticALTER(replace(SMALL, n_layers=2).to_reference_config())
    with pytest.raises(ValueError, match="does not match"):
        ALTERBackbone(SMALL, model=model)


def test_adapter_rejects_a_foreign_model() -> None:
    with pytest.raises(TypeError, match="StaticALTER"):
        ALTERBackbone(SMALL, model=torch.nn.Linear(2, 2))


def test_base_backbone_contract_is_enforced_on_bad_input() -> None:
    torch.manual_seed(7)
    backbone = ALTERBackbone(SMALL).eval()

    with pytest.raises(ValueError):
        backbone(torch.zeros(2, SMALL.n_rois + 1, SMALL.n_rois + 1))
    with pytest.raises(ValueError):
        backbone(torch.full((2, SMALL.n_rois, SMALL.n_rois), float("inf")))


def test_config_validation_is_delegated_to_the_reference_contract() -> None:
    # 20 + 4 = 24 is not divisible by 5.
    with pytest.raises(ValueError, match="divisible by nhead"):
        ALTERBackboneConfig(n_rois=20, pos_embed_dim=4, nhead=5)
    with pytest.raises(ValueError, match="pos_encoding"):
        ALTERBackboneConfig(pos_encoding="spd")


def _parameters_without_grad(module) -> list[str]:
    return [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


def test_classify_gives_every_trainable_parameter_a_gradient() -> None:
    torch.manual_seed(8)
    backbone = ALTERBackbone(SMALL)
    backbone.train()
    backbone.classify(_correlation_like(4, SMALL.n_rois, 11)).sum().backward()

    assert _parameters_without_grad(backbone) == []


def test_encode_gives_every_trainable_parameter_except_the_head_a_gradient() -> None:
    # The head is deliberately outside the encoder: the vanilla pipeline
    # re-attaches `reference_model.classifier` itself, so `encode()` leaving it
    # ungradiented is the contract, not a break in the graph.
    torch.manual_seed(9)
    backbone = ALTERBackbone(SMALL)
    backbone.train()
    backbone.encode(_correlation_like(4, SMALL.n_rois, 11)).sum().backward()

    without_grad = _parameters_without_grad(backbone)
    assert without_grad == [
        name
        for name, _ in backbone.named_parameters()
        if name.startswith("reference_model.classifier.")
    ]
    assert without_grad, "expected the head to be excluded from encode()"
