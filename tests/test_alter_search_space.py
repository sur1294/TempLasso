"""
``ALTERSearchSpace`` and the vanilla-ALTER candidate generator.

The important property is that the DEFAULT space is a singleton reproducing
the released ALTER configuration. A baseline whose hyperparameters we tuned
against a baseline whose authors' we did not is not a comparison, so widening
an axis has to be a visible decision at the call site.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from unified_bn.backbones import ALTERBackboneConfig
from unified_bn.training.config import TrainingConfig
from unified_bn.training.search_space import (
    ALTERSearchSpace,
    BQNSearchSpace,
    SearchSpaceError,
    candidate_fingerprint,
    generate_alter_backbone_configs,
    generate_vanilla_alter_candidates,
    generate_vanilla_bqn_candidates,
)


def test_default_space_is_a_singleton_released_configuration() -> None:
    configs = generate_alter_backbone_configs()
    assert len(configs) == 1

    config = configs[0]
    assert config == {
        "name": "alter",
        "n_rois": 100,
        "n_layers": 2,
        "pos_encoding": "rrwp",
        "pos_embed_dim": 32,
        "nhead": 4,
        "ffn_dim": 1024,
        "dropout": 0.1,
        "cluster_num": 100,
        "edge_threshold": 0.0,
        "transformer_bias": True,
        "orthogonal_centers": True,
        "freeze_centers": True,
        "project_assignment": True,
    }


def test_payload_keys_are_exactly_the_backbone_config_fields_plus_name() -> None:
    config = generate_alter_backbone_configs()[0]
    assert set(config) == {"name"} | {
        field.name for field in fields(ALTERBackboneConfig)
    }


def test_axes_expand_to_their_cartesian_product() -> None:
    space = ALTERSearchSpace(
        n_layers=(1, 2, 3),
        dropouts=(0.0, 0.1),
        pos_encodings=("rrwp", "none"),
        edge_thresholds=(0.0, 0.3),
    )
    configs = generate_alter_backbone_configs(space)
    assert len(configs) == 3 * 2 * 2 * 2
    assert len({tuple(sorted(c.items())) for c in configs}) == len(configs)


def test_ablation_axes_are_reachable() -> None:
    # ALTER's claim is about the long-range walk, so these two axes are the
    # ones an ablation actually needs.
    encodings = {
        c["pos_encoding"]
        for c in generate_alter_backbone_configs(
            ALTERSearchSpace(pos_encodings=("rrwp", "identity", "none"))
        )
    }
    assert encodings == {"rrwp", "identity", "none"}

    thresholds = {
        c["edge_threshold"]
        for c in generate_alter_backbone_configs(
            ALTERSearchSpace(edge_thresholds=(0.0, 0.3))
        )
    }
    assert thresholds == {0.0, 0.3}


def test_candidates_are_vanilla_and_uniquely_fingerprinted() -> None:
    space = ALTERSearchSpace(n_layers=(1, 2, 3), dropouts=(0.0, 0.1))
    candidates = generate_vanilla_alter_candidates(
        seed=42, split_identity="sha256:split42", backbone_space=space
    )

    assert len(candidates) == 6
    for candidate in candidates:
        assert candidate.identity.mode == "vanilla"
        assert candidate.identity.seed == 42
        assert candidate.identity.backbone_config["name"] == "alter"
        assert candidate.identity.temporal_config is None
        assert candidate.identity.window_policy is None
        assert candidate.identity.fusion_config is None

    fingerprints = {candidate_fingerprint(c) for c in candidates}
    assert len(fingerprints) == len(candidates)


def test_alter_and_bqn_candidates_differ_only_in_the_backbone() -> None:
    common = dict(seed=42, split_identity="sha256:split42")
    alter = generate_vanilla_alter_candidates(**common)[0]
    bqn = generate_vanilla_bqn_candidates(
        **common,
        backbone_space=BQNSearchSpace(
            n_layers=(3,), activations=("leaky_relu",), dropouts=(0.1,)
        ),
    )[0]

    assert alter.training == bqn.training
    alter_identity = alter.identity.to_dict()
    bqn_identity = bqn.identity.to_dict()
    differing = {
        key
        for key in alter_identity
        if alter_identity[key] != bqn_identity[key]
    }
    assert differing == {"backbone_config"}
    assert candidate_fingerprint(alter) != candidate_fingerprint(bqn)


def test_training_configs_multiply_the_grid() -> None:
    candidates = generate_vanilla_alter_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=ALTERSearchSpace(n_layers=(1, 2)),
        training_configs=(TrainingConfig(epochs=1), TrainingConfig(epochs=2)),
    )
    assert len(candidates) == 4


def test_unbuildable_architecture_is_rejected_at_generation_time() -> None:
    # 100 + 32 = 132 is not divisible by 7. Catching this here means a bad
    # sweep fails before any job is scheduled, not on the first forward pass.
    with pytest.raises(SearchSpaceError, match="unbuildable"):
        generate_alter_backbone_configs(ALTERSearchSpace(nhead=7))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_layers": ()},
        {"n_layers": (1, 1)},
        {"n_layers": [1, 2]},
        {"n_layers": (0,)},
        {"dropouts": (1.0,)},
        {"pos_encodings": ("spd",)},
        {"edge_thresholds": (-0.1,)},
        {"edge_thresholds": ("0.3",)},
        {"n_rois": 0},
        {"pos_embed_dim": 0},
        {"cluster_num": 0},
        {"ffn_dim": 0},
        {"freeze_centers": "yes"},
    ],
)
def test_search_space_fails_closed(kwargs) -> None:
    with pytest.raises((TypeError, SearchSpaceError, ValueError)):
        ALTERSearchSpace(**kwargs)


def test_generator_rejects_a_foreign_space() -> None:
    with pytest.raises(TypeError, match="ALTERSearchSpace"):
        generate_alter_backbone_configs(BQNSearchSpace())
