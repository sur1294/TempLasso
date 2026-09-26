"""
Model-factory dispatch for ``backbone name='alter'``.

Two things are being pinned here. First, that a vanilla ALTER candidate builds
the released model and nothing else -- same policy as the BQN vanilla path,
where the pipeline reuses the reference head rather than initialising a new
one. Second, that ALTER slots into the temporal and fusion pipelines through
exactly the same dispatch as BQN (TempALTER, ALTER+TempALTER, ALTER+TempLasso),
and that moving the X+TempX candidate out of the CLI kept its fingerprint.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from unified_bn.backbones import ALTERBackbone, BQNBackbone
from unified_bn.pipelines.fusion import FusionPipeline
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.pipelines.vanilla import VanillaPipeline
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.config import CandidateRunConfig, RunIdentity
from unified_bn.training.model_factory import (
    ModelFactoryError,
    build_model_from_candidate,
)
from unified_bn.training.search_space import (
    ALTERSearchSpace,
    BQNSearchSpace,
    BiGRUSearchSpace,
    LassoEdgeSearchSpace,
    candidate_fingerprint,
    generate_alter_backbone_configs,
    generate_bigru_temporal_configs,
    generate_bqn_backbone_configs,
    generate_fusion_candidates,
    generate_fusion_templasso_candidates,
    generate_temporal_bqn_bigru_candidates,
    generate_vanilla_alter_candidates,
    generate_vanilla_bqn_candidates,
)


SMALL_SPACE = ALTERSearchSpace(
    n_layers=(1,), n_rois=20, pos_embed_dim=4, nhead=4, ffn_dim=16,
    cluster_num=3,
)


def _alter_candidate(space: ALTERSearchSpace = SMALL_SPACE):
    return generate_vanilla_alter_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=space,
    )[0]


def _correlation_like(batch_size: int, n_rois: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, n_rois, n_rois, generator=generator)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)
    diagonal = torch.arange(n_rois)
    x[:, diagonal, diagonal] = 1.0
    return x


def test_vanilla_alter_candidate_builds_the_reference_path() -> None:
    torch.manual_seed(42)
    model = build_model_from_candidate(_alter_candidate())

    assert isinstance(model, VanillaPipeline)
    assert isinstance(model.backbone, ALTERBackbone)
    assert model.backbone.config.n_layers == 1
    assert model.embedding_dim == 8 * SMALL_SPACE.cluster_num

    # No replacement classifier: literal module identity, exactly as the BQN
    # vanilla path asserts in test_training_stage6j_model_factory.py.
    assert model.classifier is model.backbone.reference_model.classifier


def test_vanilla_alter_creates_no_parameters_outside_the_reference_model() -> None:
    torch.manual_seed(43)
    model = build_model_from_candidate(_alter_candidate())

    pipeline_parameter_ids = {id(p) for p in model.parameters()}
    reference_parameter_ids = {
        id(p) for p in model.backbone.reference_model.parameters()
    }
    assert pipeline_parameter_ids == reference_parameter_ids


def test_vanilla_alter_logits_are_bitwise_reference_logits() -> None:
    torch.manual_seed(44)
    model = build_model_from_candidate(_alter_candidate())
    model.eval()

    fc = _correlation_like(3, SMALL_SPACE.n_rois, 13)
    with torch.no_grad():
        unified = model(fc).logits
        reference = model.backbone.reference_model(fc)

    torch.testing.assert_close(unified, reference, rtol=0.0, atol=0.0)


def test_full_size_alter_candidate_has_the_released_shape() -> None:
    torch.manual_seed(45)
    model = build_model_from_candidate(_alter_candidate(ALTERSearchSpace()))
    model.eval()

    assert model.embedding_dim == 800
    with torch.no_grad():
        assert model(_correlation_like(2, 100, 17)).logits.shape == (2, 2)


def test_default_bqn_path_is_untouched() -> None:
    torch.manual_seed(46)
    candidate = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(3,), activations=("elu",), dropouts=(0.3,)
        ),
    )[0]
    model = build_model_from_candidate(candidate)
    assert isinstance(model.backbone, BQNBackbone)


def test_factory_rejects_unknown_alter_config_key() -> None:
    candidate = _alter_candidate()
    bad_backbone = dict(candidate.identity.backbone_config)
    bad_backbone["mystery"] = 123

    bad = replace(
        candidate,
        identity=replace(candidate.identity, backbone_config=bad_backbone),
    )

    with pytest.raises(ModelFactoryError, match="unknown"):
        build_model_from_candidate(bad)


def test_factory_rejects_missing_alter_config_key() -> None:
    candidate = _alter_candidate()
    bad_backbone = dict(candidate.identity.backbone_config)
    bad_backbone.pop("cluster_num")

    bad = replace(
        candidate,
        identity=replace(candidate.identity, backbone_config=bad_backbone),
    )

    with pytest.raises(ModelFactoryError, match="missing"):
        build_model_from_candidate(bad)


def test_factory_rejects_an_architecturally_impossible_alter_config() -> None:
    candidate = _alter_candidate()
    bad_backbone = dict(candidate.identity.backbone_config)
    bad_backbone["nhead"] = 7  # 24 % 7 != 0

    bad = replace(
        candidate,
        identity=replace(candidate.identity, backbone_config=bad_backbone),
    )

    with pytest.raises(ModelFactoryError, match="Invalid ALTER"):
        build_model_from_candidate(bad)


def test_unknown_backbone_error_still_names_bqn() -> None:
    # test_training_stage6j_model_factory.py matches on "backbone name='bqn'";
    # extending the message must not break that contract.
    candidate = _alter_candidate()
    bad_backbone = dict(candidate.identity.backbone_config)
    bad_backbone["name"] = "unregistered_backbone"

    bad = replace(
        candidate,
        identity=replace(candidate.identity, backbone_config=bad_backbone),
    )

    with pytest.raises(ModelFactoryError, match="backbone name='bqn'"):
        build_model_from_candidate(bad)


def _temporal_candidate_with_alter_backbone():
    reference = generate_temporal_bqn_bigru_candidates(
        seed=42,
        split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        backbone_space=BQNSearchSpace(
            n_layers=(1,), activations=("elu",), dropouts=(0.0,), n_rois=20,
            cluster_num=2,
        ),
        temporal_space=BiGRUSearchSpace(
            hidden_dims=(8,), num_layers=(1,), poolings=("final",)
        ),
    )[0]
    return replace(
        reference,
        identity=replace(
            reference.identity,
            backbone_config=generate_alter_backbone_configs(SMALL_SPACE)[0],
        ),
    )


def _dynamic_like(batch_size: int, k: int, n_rois: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, k, n_rois, n_rois, generator=generator)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)
    diagonal = torch.arange(n_rois)
    x[..., diagonal, diagonal] = 1.0
    return x


def test_temporal_candidate_with_alter_backbone_builds_temp_alter() -> None:
    # TempALTER: the same ALTER module encodes every valid window, BiGRU
    # aggregates. Nothing ALTER-specific lives in TemporalPipeline.
    torch.manual_seed(47)
    model = build_model_from_candidate(_temporal_candidate_with_alter_backbone())

    assert isinstance(model, TemporalPipeline)
    assert isinstance(model.backbone, ALTERBackbone)
    assert model.temporal_aggregator.input_dim == model.backbone.embedding_dim
    assert isinstance(model.classifier, nn.Linear)

    model.eval()
    lengths = torch.tensor([3, 2], dtype=torch.long)
    dynamic_fc = _dynamic_like(2, 3, SMALL_SPACE.n_rois, 19)
    dynamic_fc[1, 2] = 1e6  # padding must never reach the backbone
    with torch.no_grad():
        output = model(dynamic_fc, lengths, return_aux=True)
    assert output.logits.shape == (2, 2)
    assert output.aux["n_valid_windows"] == 5


def _fusion_identity(*, static_config, temporal_backbone_config=None):
    return RunIdentity(
        mode="fusion",
        seed=42,
        split_identity="sha256:split42",
        backbone_config=static_config,
        temporal_config={"name": "bigru", "hidden_dim": 8, "num_layers": 1,
                         "dropout": 0.0, "pooling": "final",
                         "projection_dim": None, "layer_norm": False},
        window_policy=WindowPolicyConfig(policy="native"),
        fusion_config={"name": "concat"},
        temporal_backbone_config=temporal_backbone_config,
    )


def test_fusion_with_alter_on_both_branches_builds_alter_plus_temp_alter() -> None:
    torch.manual_seed(48)
    candidate = CandidateRunConfig(
        identity=_fusion_identity(
            static_config=generate_alter_backbone_configs(SMALL_SPACE)[0]
        )
    )
    model = build_model_from_candidate(candidate)

    assert isinstance(model, FusionPipeline)
    assert isinstance(model.static_backbone, ALTERBackbone)
    assert isinstance(model.temporal_backbone, ALTERBackbone)
    # Two independent module instances with independent weights, exactly as
    # the historical BQN+BQN fusion baseline is built.
    assert model.static_backbone is not model.temporal_backbone
    static_ids = {id(p) for p in model.static_backbone.parameters()}
    temporal_ids = {id(p) for p in model.temporal_backbone.parameters()}
    assert not static_ids & temporal_ids
    assert model.fusion.output_dim == model.static_backbone.embedding_dim + 16


def test_generate_fusion_candidates_matches_the_historical_inline_identity() -> None:
    # The X+TempX candidate used to be assembled inline in the CLI. Moving it
    # into search_space must not change any historical BQN+BQN fingerprint,
    # so temporal_backbone_config stays unset and to_dict() is byte-identical.
    space = BQNSearchSpace(n_layers=(1,), activations=("elu",), dropouts=(0.0,))
    temporal_space = BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",))
    candidate = generate_fusion_candidates(
        seed=42,
        split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=temporal_space,
        backbone_space=space,
    )[0]
    inline = CandidateRunConfig(
        identity=RunIdentity(
            mode="fusion",
            seed=42,
            split_identity="sha256:split42",
            backbone_config=generate_bqn_backbone_configs(space)[0],
            temporal_config=generate_bigru_temporal_configs(temporal_space)[0],
            window_policy=WindowPolicyConfig(policy="native"),
            fusion_config={"name": "concat"},
        )
    )
    assert candidate.identity.temporal_backbone_config is None
    assert candidate.to_dict() == inline.to_dict()
    assert candidate_fingerprint(candidate) == candidate_fingerprint(inline)


def test_fusion_templasso_accepts_an_alter_static_branch() -> None:
    candidate = generate_fusion_templasso_candidates(
        seed=42,
        split_identity="sha256:split42",
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",)),
        backbone_space=SMALL_SPACE,
        selector_space=LassoEdgeSearchSpace(Cs=(0.01,), n_rois=SMALL_SPACE.n_rois),
    )[0]
    assert candidate.identity.backbone_config["name"] == "alter"
    assert candidate.identity.temporal_backbone_config["name"] == "lasso_edges"


def test_factory_signature_and_purity_are_unchanged() -> None:
    signature = inspect.signature(build_model_from_candidate)
    assert tuple(signature.parameters) == ("run_config",)

    source = inspect.getsource(build_model_from_candidate).lower()
    for token in (
        "dataloader",
        "h5py",
        "test_batches",
        "test_loader",
        "train_candidate(",
        "evaluate_final_test",
    ):
        assert token not in source
