from __future__ import annotations

import pytest

from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.config import OptimizerConfig, TrainingConfig
from unified_bn.training.search_space import (
    BQNSearchSpace,
    BiGRUSearchSpace,
    SearchSpaceError,
    candidate_fingerprint,
    generate_bigru_temporal_configs,
    generate_bqn_backbone_configs,
    generate_temporal_bqn_bigru_candidates,
    generate_vanilla_bqn_candidates,
)


def _assert_no_test_keys(payload) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert "test" not in str(key).lower()
            _assert_no_test_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_no_test_keys(value)


def test_default_bqn_space_is_exact_60_candidate_atlas_grid() -> None:
    configs = generate_bqn_backbone_configs()
    assert len(configs) == 5 * 3 * 4
    assert len({(c["n_layers"], c["activation"], c["dropout"]) for c in configs}) == 60
    assert {c["n_layers"] for c in configs} == {1, 2, 3, 4, 5}
    assert {c["activation"] for c in configs} == {"gelu", "leaky_relu", "elu"}
    assert {c["dropout"] for c in configs} == {0.0, 0.1, 0.2, 0.3}
    for cfg in configs:
        assert cfg["name"] == "bqn"
        assert cfg["n_rois"] == 100
        assert cfg["cluster_num"] == 4
        assert cfg["preserve_unused_mlp_b"] is True
        assert cfg["orthogonal_centers"] is True
        assert cfg["freeze_centers"] is True
        assert cfg["project_assignment"] is True


def test_bqn_generation_is_deterministic() -> None:
    assert generate_bqn_backbone_configs() == generate_bqn_backbone_configs()


def test_duplicate_bqn_axis_value_is_rejected() -> None:
    with pytest.raises(SearchSpaceError, match="duplicate"):
        BQNSearchSpace(n_layers=(1, 1), activations=("elu",), dropouts=(0.0,))


def test_bigru_one_layer_dropout_is_canonical_zero() -> None:
    space = BiGRUSearchSpace(
        hidden_dims=(32,),
        num_layers=(1, 2),
        poolings=("final",),
        stacked_dropouts=(0.0, 0.1, 0.3),
    )
    configs = generate_bigru_temporal_configs(space)
    one_layer = [c for c in configs if c["num_layers"] == 1]
    two_layer = [c for c in configs if c["num_layers"] == 2]
    assert len(one_layer) == 1
    assert one_layer[0]["dropout"] == 0.0
    assert {c["dropout"] for c in two_layer} == {0.0, 0.1, 0.3}


def test_completely_inactive_one_layer_dropout_space_is_rejected() -> None:
    with pytest.raises(SearchSpaceError, match="inactive"):
        BiGRUSearchSpace(
            hidden_dims=(32,),
            num_layers=(1,),
            poolings=("final",),
            stacked_dropouts=(0.0, 0.2),
        )


def test_attention_parameters_exist_only_for_attention_pooling() -> None:
    space = BiGRUSearchSpace(
        hidden_dims=(32,),
        num_layers=(1,),
        poolings=("final", "mean", "attention"),
        stacked_dropouts=(0.0,),
        attention_hidden_dims=(16, 32),
        attention_dropouts=(0.0, 0.2),
    )
    configs = generate_bigru_temporal_configs(space)
    non_attention = [c for c in configs if c["pooling"] in {"final", "mean"}]
    attention = [c for c in configs if c["pooling"] == "attention"]
    assert len(non_attention) == 2
    assert len(attention) == 4
    for cfg in non_attention:
        assert "attention_hidden_dim" not in cfg
        assert "attention_dropout" not in cfg
    assert {(c["attention_hidden_dim"], c["attention_dropout"]) for c in attention} == {
        (16, 0.0), (16, 0.2), (32, 0.0), (32, 0.2)
    }


def test_bigru_conditional_grid_has_expected_unique_count() -> None:
    space = BiGRUSearchSpace(
        hidden_dims=(32,),
        num_layers=(1, 2),
        poolings=("final", "attention"),
        stacked_dropouts=(0.0, 0.2),
        attention_hidden_dims=(16, 32),
        attention_dropouts=(0.0, 0.1),
        projection_dims=(None, 16),
        layer_norms=(False, True),
    )
    configs = generate_bigru_temporal_configs(space)
    assert len(configs) == 60
    assert len({str(sorted(c.items())) for c in configs}) == 60


def test_vanilla_candidates_are_config_only_and_test_free() -> None:
    candidates = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(1, 2), activations=("elu",), dropouts=(0.0, 0.1)
        ),
    )
    assert len(candidates) == 4
    for candidate in candidates:
        assert candidate.identity.mode == "vanilla"
        assert candidate.identity.seed == 42
        assert candidate.identity.temporal_config is None
        assert candidate.identity.window_policy is None
        assert candidate.training.checkpoint_selection.metric == "val_loss"
        assert candidate.training.checkpoint_selection.mode == "min"
        _assert_no_test_keys(candidate.to_dict())


def test_temporal_candidates_preserve_explicit_native_window_policy() -> None:
    temporal_space = BiGRUSearchSpace(
        hidden_dims=(32,),
        num_layers=(1,),
        poolings=("final", "mean"),
        stacked_dropouts=(0.0,),
    )
    window_policy = WindowPolicyConfig(policy="native")
    candidates = generate_temporal_bqn_bigru_candidates(
        seed=1042,
        split_identity="sha256:split1042",
        window_policy=window_policy,
        temporal_space=temporal_space,
        backbone_space=BQNSearchSpace(
            n_layers=(3,), activations=("elu",), dropouts=(0.1,)
        ),
    )
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.identity.mode == "temporal"
        assert candidate.identity.window_policy == window_policy
        assert candidate.identity.backbone_config["name"] == "bqn"
        assert candidate.identity.temporal_config["name"] == "bigru"
        assert "input_dim" not in candidate.identity.temporal_config
        _assert_no_test_keys(candidate.to_dict())


def test_training_config_is_an_independent_search_axis() -> None:
    training_a = TrainingConfig(
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=1e-4)
    )
    training_b = TrainingConfig(
        optimizer=OptimizerConfig(lr=3e-4, weight_decay=1e-4)
    )
    candidates = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(3,), activations=("elu",), dropouts=(0.1,)
        ),
        training_configs=(training_a, training_b),
    )
    assert len(candidates) == 2
    assert {c.training.optimizer.lr for c in candidates} == {1e-4, 3e-4}


def test_duplicate_training_configs_are_rejected() -> None:
    training = TrainingConfig()
    with pytest.raises(SearchSpaceError, match="duplicate"):
        generate_vanilla_bqn_candidates(
            seed=42,
            split_identity="sha256:split42",
            backbone_space=BQNSearchSpace(
                n_layers=(3,), activations=("elu",), dropouts=(0.1,)
            ),
            training_configs=(training, training),
        )


def test_candidate_fingerprint_is_stable_and_configuration_sensitive() -> None:
    candidate_a = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(3,), activations=("elu",), dropouts=(0.1,)
        ),
    )[0]
    candidate_b = generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(4,), activations=("elu",), dropouts=(0.1,)
        ),
    )[0]
    assert candidate_fingerprint(candidate_a) == candidate_fingerprint(candidate_a)
    assert candidate_fingerprint(candidate_a) != candidate_fingerprint(candidate_b)
    assert len(candidate_fingerprint(candidate_a)) == 64


def test_temporal_generation_is_deterministic_and_duplicate_free() -> None:
    kwargs = dict(
        seed=2042,
        split_identity="sha256:split2042",
        window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BiGRUSearchSpace(
            hidden_dims=(32, 64),
            num_layers=(1, 2),
            poolings=("final", "attention"),
            stacked_dropouts=(0.0, 0.2),
            attention_hidden_dims=(16,),
            attention_dropouts=(0.0,),
            projection_dims=(None,),
            layer_norms=(False, True),
        ),
        backbone_space=BQNSearchSpace(
            n_layers=(3,), activations=("gelu",), dropouts=(0.2,)
        ),
    )
    first = generate_temporal_bqn_bigru_candidates(**kwargs)
    second = generate_temporal_bqn_bigru_candidates(**kwargs)
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
    fingerprints = [candidate_fingerprint(c) for c in first]
    assert len(fingerprints) == len(set(fingerprints))
