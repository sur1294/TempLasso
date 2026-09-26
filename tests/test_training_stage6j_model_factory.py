from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from unified_bn.backbones.bqn import BQNBackbone
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.pipelines.vanilla import VanillaPipeline
from unified_bn.temporal.bigru import BiGRUAggregator
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.model_factory import (
    ModelFactoryError,
    build_model_from_candidate,
)
from unified_bn.training.search_space import (
    BQNSearchSpace,
    BiGRUSearchSpace,
    generate_temporal_bqn_bigru_candidates,
    generate_vanilla_bqn_candidates,
)


def _vanilla_candidate():
    return generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(3,),
            activations=("elu",),
            dropouts=(0.3,),
        ),
    )[0]


def _temporal_candidate(
    *,
    pooling="final",
    projection_dim=None,
    layer_norm=False,
):
    return generate_temporal_bqn_bigru_candidates(
        seed=42,
        split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        backbone_space=BQNSearchSpace(
            n_layers=(3,),
            activations=("elu",),
            dropouts=(0.3,),
        ),
        temporal_space=BiGRUSearchSpace(
            hidden_dims=(32,),
            num_layers=(1,),
            poolings=(pooling,),
            stacked_dropouts=(0.0,),
            attention_hidden_dims=(16,),
            attention_dropouts=(0.0,),
            projection_dims=(projection_dim,),
            layer_norms=(layer_norm,),
        ),
    )[0]


def _correlation_like(batch_size: int) -> torch.Tensor:
    x = torch.randn(batch_size, 100, 100)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)
    diagonal = torch.arange(100)
    x[:, diagonal, diagonal] = 1.0
    return x


def test_vanilla_factory_builds_exact_reference_bqn_path() -> None:
    torch.manual_seed(42)
    candidate = _vanilla_candidate()
    model = build_model_from_candidate(candidate)

    assert isinstance(model, VanillaPipeline)
    assert isinstance(model.backbone, BQNBackbone)

    assert model.backbone.config.n_layers == 3
    assert model.backbone.config.activation == "elu"
    assert model.backbone.config.dropout == pytest.approx(0.3)

    # No replacement classifier: literal module identity.
    assert (
        model.classifier
        is model.backbone.reference_model.classifier
    )


def test_vanilla_factory_logits_are_bitwise_reference_bqn_logits() -> None:
    torch.manual_seed(7)
    candidate = _vanilla_candidate()
    model = build_model_from_candidate(candidate)
    model.eval()

    fc = _correlation_like(3)

    with torch.no_grad():
        unified = model(fc).logits
        reference = model.backbone.reference_model(fc)

    torch.testing.assert_close(
        unified,
        reference,
        rtol=0.0,
        atol=0.0,
    )


def test_vanilla_factory_does_not_create_independent_classifier_parameters() -> None:
    candidate = _vanilla_candidate()
    model = build_model_from_candidate(candidate)

    pipeline_parameter_ids = {id(parameter) for parameter in model.parameters()}
    reference_parameter_ids = {
        id(parameter)
        for parameter in model.backbone.reference_model.parameters()
    }

    assert pipeline_parameter_ids == reference_parameter_ids


def test_temporal_factory_builds_shared_bqn_bigru_pipeline() -> None:
    candidate = _temporal_candidate()
    model = build_model_from_candidate(candidate)

    assert isinstance(model, TemporalPipeline)
    assert isinstance(model.backbone, BQNBackbone)
    assert isinstance(model.temporal_aggregator, BiGRUAggregator)

    assert model.temporal_aggregator.input_dim == model.backbone.embedding_dim
    assert model.temporal_aggregator.config.hidden_dim == 32
    assert model.temporal_aggregator.config.num_layers == 1
    assert model.temporal_aggregator.config.dropout == pytest.approx(0.0)
    assert model.temporal_aggregator.config.pooling == "final"

    assert model.window_policy == candidate.identity.window_policy

    # Current Stage-6 temporal baseline uses the minimal pipeline-level
    # classifier. Future classifier families must become explicit candidate
    # identity before they are tuned.
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.in_features == model.temporal_aggregator.output_dim
    assert model.classifier.out_features == 2


def test_temporal_factory_derives_bigru_input_dim_from_backbone() -> None:
    candidate = _temporal_candidate(
        pooling="mean",
        projection_dim=24,
        layer_norm=True,
    )
    model = build_model_from_candidate(candidate)

    assert model.backbone.embedding_dim == 32
    assert model.temporal_aggregator.config.input_dim == 32
    assert model.temporal_aggregator.config.projection_dim == 24
    assert model.temporal_aggregator.output_dim == 24
    assert model.temporal_aggregator.config.layer_norm is True
    assert model.classifier.in_features == 24


def test_attention_candidate_rehydrates_attention_only_configuration() -> None:
    candidate = generate_temporal_bqn_bigru_candidates(
        seed=1042,
        split_identity="sha256:split1042",
        window_policy=WindowPolicyConfig(policy="native"),
        backbone_space=BQNSearchSpace(
            n_layers=(4,),
            activations=("gelu",),
            dropouts=(0.3,),
        ),
        temporal_space=BiGRUSearchSpace(
            hidden_dims=(64,),
            num_layers=(2,),
            poolings=("attention",),
            stacked_dropouts=(0.2,),
            attention_hidden_dims=(16,),
            attention_dropouts=(0.1,),
            projection_dims=(32,),
            layer_norms=(True,),
        ),
    )[0]

    model = build_model_from_candidate(candidate)

    cfg = model.temporal_aggregator.config
    assert cfg.input_dim == 32
    assert cfg.hidden_dim == 64
    assert cfg.num_layers == 2
    assert cfg.dropout == pytest.approx(0.2)
    assert cfg.pooling == "attention"
    assert cfg.attention_hidden_dim == 16
    assert cfg.attention_dropout == pytest.approx(0.1)
    assert cfg.projection_dim == 32
    assert cfg.layer_norm is True


def test_final_pooling_rehydrates_inactive_attention_defaults_only() -> None:
    candidate = _temporal_candidate(pooling="final")

    # Stage 6H intentionally removes attention-only keys from final pooling.
    assert "attention_hidden_dim" not in candidate.identity.temporal_config
    assert "attention_dropout" not in candidate.identity.temporal_config

    model = build_model_from_candidate(candidate)
    cfg = model.temporal_aggregator.config

    assert cfg.attention_hidden_dim == 32
    assert cfg.attention_dropout == pytest.approx(0.0)


def test_temporal_forward_shapes_from_factory() -> None:
    torch.manual_seed(11)
    candidate = _temporal_candidate()
    model = build_model_from_candidate(candidate)
    model.eval()

    lengths = torch.tensor([3, 2], dtype=torch.long)
    dynamic_fc = torch.zeros(2, 3, 100, 100)

    first = _correlation_like(3)
    second = _correlation_like(2)

    dynamic_fc[0, :3] = first
    dynamic_fc[1, :2] = second
    dynamic_fc[1, 2] = 1e6  # padding must not enter the backbone

    with torch.no_grad():
        output = model(
            dynamic_fc,
            lengths,
            return_aux=True,
        )

    assert output.logits.shape == (2, 2)
    assert output.temporal_embedding.shape == (2, 64)
    assert output.aux["n_valid_windows"] == 5


def test_factory_rejects_unknown_backbone_name() -> None:
    candidate = _vanilla_candidate()
    bad_backbone = dict(candidate.identity.backbone_config)
    bad_backbone["name"] = "unregistered_backbone"

    bad = replace(
        candidate,
        identity=replace(
            candidate.identity,
            backbone_config=bad_backbone,
        ),
    )

    with pytest.raises(ModelFactoryError, match="backbone name='bqn'"):
        build_model_from_candidate(bad)


def test_factory_rejects_unknown_bqn_config_key() -> None:
    candidate = _vanilla_candidate()
    bad_backbone = dict(candidate.identity.backbone_config)
    bad_backbone["mystery"] = 123

    bad = replace(
        candidate,
        identity=replace(
            candidate.identity,
            backbone_config=bad_backbone,
        ),
    )

    with pytest.raises(ModelFactoryError, match="unknown"):
        build_model_from_candidate(bad)


def test_factory_rejects_unknown_temporal_name() -> None:
    candidate = _temporal_candidate()
    bad_temporal = dict(candidate.identity.temporal_config)
    bad_temporal["name"] = "transformer"

    bad = replace(
        candidate,
        identity=replace(
            candidate.identity,
            temporal_config=bad_temporal,
        ),
    )

    with pytest.raises(ModelFactoryError, match="temporal name='bigru'"):
        build_model_from_candidate(bad)


def test_factory_rejects_candidate_supplied_bigru_input_dim() -> None:
    candidate = _temporal_candidate()
    bad_temporal = dict(candidate.identity.temporal_config)
    bad_temporal["input_dim"] = 999

    bad = replace(
        candidate,
        identity=replace(
            candidate.identity,
            temporal_config=bad_temporal,
        ),
    )

    with pytest.raises(ModelFactoryError, match="input_dim is derived"):
        build_model_from_candidate(bad)


def test_factory_is_model_construction_only() -> None:
    import inspect

    signature = inspect.signature(build_model_from_candidate)

    assert tuple(signature.parameters) == ("run_config",)

    source = inspect.getsource(build_model_from_candidate).lower()
    forbidden = (
        "dataloader",
        "h5py",
        "test_batches",
        "test_loader",
        "train_candidate(",
        "evaluate_final_test",
    )
    for token in forbidden:
        assert token not in source
