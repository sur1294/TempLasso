"""
``--backbone alter`` on the real-ABIDE CLI.

Two invariants matter more than the new flags themselves:

  * every historical invocation keeps its exact meaning, because ``--backbone``
    defaults to ``bqn`` and, for ALTER, the ``paper`` variant with no flags
    resolves to the released values;
  * the argument groups stay mutually exclusive in both directions, so a run
    can never silently ignore a flag the operator believed was in effect.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import unified_bn.training.abide_cli as cli_module
from unified_bn.training.abide_cli import (
    ABIDECLIError,
    build_parser,
    build_single_candidate,
    run_from_args,
)


def _args(tmp_path: Path, *extra: str, mode: str = "vanilla") -> Namespace:
    argv = [
        "--mode", mode,
        "--seed", "42",
        "--split-path", str(tmp_path / "split.npz"),
        "--static-h5", str(tmp_path / "static.h5"),
        "--temporal-h5", str(tmp_path / "temporal.h5"),
        "--search-dir", str(tmp_path / "search"),
        "--device", "cpu",
        *extra,
    ]
    return build_parser().parse_args(argv)


BQN_ARGS = ("--bqn-layers", "3", "--bqn-activation", "elu", "--bqn-dropout", "0.3")
TEMPORAL_ARGS = (
    "--bigru-hidden-dim", "32", "--bigru-layers", "1", "--bigru-pooling", "final",
)


# ------------------------------------------------------- parser defaults
def test_backbone_defaults_to_bqn_and_alter_flags_resolve_to_released_values(
    tmp_path,
) -> None:
    args = _args(tmp_path, *BQN_ARGS)
    assert args.backbone == "bqn"
    assert args.alter_variant == "paper"
    # Flags default to None so "explicitly set" is unambiguous...
    for name in cli_module.ALTER_CLI_DEFAULTS:
        assert getattr(args, name) is None, name
    # ...and resolve to the released configuration.
    resolved = cli_module.resolve_alter_arguments(args)
    assert resolved == {
        "alter_layers": 2,
        "alter_dropout": 0.1,
        "alter_pos_encoding": "rrwp",
        "alter_pe_dim": 32,
        "alter_heads": 4,
        "alter_ffn_dim": 1024,
        "alter_cluster_num": 100,
        "alter_edge_threshold": 0.0,
        "alter_transformer_bias": True,
    }


def test_alter_cli_defaults_table_is_the_paper_variant(tmp_path) -> None:
    args = _args(tmp_path, *BQN_ARGS)
    assert cli_module.resolve_alter_arguments(args) == cli_module.ALTER_CLI_DEFAULTS
    assert cli_module.ALTER_VARIANTS["paper"] == {}


def test_harness_variant_resolves_to_the_collaborators_model(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, "--backbone", "alter", "--alter-variant", "harness"),
        split_identity="sha256:split42",
    )
    backbone = candidate.identity.backbone_config
    assert backbone["nhead"] == 1
    assert backbone["edge_threshold"] == pytest.approx(0.3)
    assert backbone["transformer_bias"] is False
    # everything else stays at the released values
    assert backbone["n_layers"] == 2
    assert backbone["pos_embed_dim"] == 32
    assert backbone["ffn_dim"] == 1024
    assert backbone["cluster_num"] == 100


def test_explicit_flag_overrides_the_variant(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, "--backbone", "alter", "--alter-variant", "harness",
              "--alter-heads", "4"),
        split_identity="sha256:split42",
    )
    backbone = candidate.identity.backbone_config
    assert backbone["nhead"] == 4                       # overridden
    assert backbone["edge_threshold"] == pytest.approx(0.3)  # from variant
    assert backbone["transformer_bias"] is False        # from variant


def test_transformer_bias_flag_parses_booleans(tmp_path) -> None:
    for text, expected in (("false", False), ("true", True), ("0", False), ("1", True)):
        candidate = build_single_candidate(
            _args(tmp_path, "--backbone", "alter", "--alter-transformer-bias", text),
            split_identity="sha256:split42",
        )
        assert candidate.identity.backbone_config["transformer_bias"] is expected
    with pytest.raises(SystemExit):
        _args(tmp_path, "--backbone", "alter", "--alter-transformer-bias", "maybe")


def test_paper_and_harness_variants_have_distinct_fingerprints(tmp_path) -> None:
    from unified_bn.training.search_space import candidate_fingerprint

    paper = build_single_candidate(
        _args(tmp_path, "--backbone", "alter"), split_identity="sha256:split42"
    )
    harness = build_single_candidate(
        _args(tmp_path, "--backbone", "alter", "--alter-variant", "harness"),
        split_identity="sha256:split42",
    )
    assert candidate_fingerprint(paper) != candidate_fingerprint(harness)


def test_parser_still_exposes_no_test_evaluation_argument() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert all("test" not in option.lower() for option in options)


def test_historical_bqn_invocation_is_unchanged(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, *BQN_ARGS), split_identity="sha256:split42"
    )
    assert candidate.identity.backbone_config["name"] == "bqn"
    assert candidate.identity.mode == "vanilla"


# ------------------------------------------------------- alter candidate
def test_alter_backbone_builds_one_released_vanilla_candidate(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, "--backbone", "alter"),
        split_identity="sha256:split42",
    )

    assert candidate.identity.mode == "vanilla"
    assert candidate.identity.backbone_config == {
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


def test_alter_flags_reach_the_candidate(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(
            tmp_path,
            "--backbone", "alter",
            "--alter-layers", "3",
            "--alter-dropout", "0.2",
            "--alter-pos-encoding", "none",
            "--alter-edge-threshold", "0.3",
            "--alter-cluster-num", "50",
        ),
        split_identity="sha256:split42",
    )

    backbone = candidate.identity.backbone_config
    assert backbone["n_layers"] == 3
    assert backbone["dropout"] == pytest.approx(0.2)
    assert backbone["pos_encoding"] == "none"
    assert backbone["edge_threshold"] == pytest.approx(0.3)
    assert backbone["cluster_num"] == 50


def test_alter_honours_expected_n_rois(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(
            tmp_path,
            "--backbone", "alter",
            "--expected-n-rois", "20",
            "--alter-pe-dim", "4",
        ),
        split_identity="sha256:split42",
    )
    assert candidate.identity.backbone_config["n_rois"] == 20
    assert candidate.identity.backbone_config["pos_embed_dim"] == 4


# ------------------------------------------------------- every X-mode takes alter
def test_temporal_mode_with_alter_builds_temp_alter(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, "--backbone", "alter", "--alter-variant", "harness",
              *TEMPORAL_ARGS, mode="temporal"),
        split_identity="sha256:split42",
    )
    assert candidate.identity.mode == "temporal"
    assert candidate.identity.backbone_config["name"] == "alter"
    assert candidate.identity.backbone_config["nhead"] == 1
    assert candidate.identity.temporal_config["name"] == "bigru"
    assert candidate.identity.temporal_config["hidden_dim"] == 32
    assert candidate.identity.window_policy.policy == "native"


def test_fusion_mode_with_alter_builds_alter_plus_temp_alter(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, "--backbone", "alter", *TEMPORAL_ARGS, mode="fusion"),
        split_identity="sha256:split42",
    )
    assert candidate.identity.mode == "fusion"
    assert candidate.identity.backbone_config["name"] == "alter"
    assert candidate.identity.fusion_config == {"name": "concat"}
    # both branches from backbone_config: no temporal_backbone_config
    assert candidate.identity.temporal_backbone_config is None


def test_fusion_templasso_mode_with_alter_builds_alter_plus_temp_lasso(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, "--backbone", "alter", *TEMPORAL_ARGS, "--lasso-C", "0.01",
              mode="fusion_templasso"),
        split_identity="sha256:split42",
    )
    assert candidate.identity.mode == "fusion"
    assert candidate.identity.backbone_config["name"] == "alter"
    assert candidate.identity.temporal_backbone_config["name"] == "lasso_edges"
    assert candidate.identity.temporal_backbone_config["C"] == pytest.approx(0.01)


def test_historical_bqn_fusion_invocation_is_unchanged(tmp_path) -> None:
    candidate = build_single_candidate(
        _args(tmp_path, *BQN_ARGS, *TEMPORAL_ARGS, mode="fusion"),
        split_identity="sha256:split42",
    )
    assert candidate.identity.backbone_config["name"] == "bqn"
    assert candidate.identity.temporal_backbone_config is None
    assert candidate.identity.fusion_config == {"name": "concat"}


def test_templasso_has_no_backbone_and_rejects_backbone_alter(tmp_path) -> None:
    args = _args(tmp_path, "--backbone", "alter", *TEMPORAL_ARGS, "--lasso-C", "0.01",
                 mode="templasso")
    with pytest.raises(ABIDECLIError, match="no static backbone"):
        build_single_candidate(args, split_identity="sha256:split42")


# ------------------------------------------------------- exclusivity rules


def test_alter_backbone_rejects_bqn_arguments(tmp_path) -> None:
    args = _args(tmp_path, "--backbone", "alter", *BQN_ARGS)
    with pytest.raises(ABIDECLIError, match="no BQN backbone"):
        build_single_candidate(args, split_identity="sha256:split42")


def test_alter_backbone_rejects_bqn_cluster_num(tmp_path) -> None:
    args = _args(tmp_path, "--backbone", "alter", "--bqn-cluster-num", "8")
    with pytest.raises(ABIDECLIError, match="--bqn-cluster-num"):
        build_single_candidate(args, split_identity="sha256:split42")


def test_alter_backbone_rejects_temporal_arguments(tmp_path) -> None:
    args = _args(tmp_path, "--backbone", "alter", *TEMPORAL_ARGS)
    with pytest.raises(ABIDECLIError, match="temporal-model arguments"):
        build_single_candidate(args, split_identity="sha256:split42")


def test_alter_backbone_rejects_lasso_arguments(tmp_path) -> None:
    args = _args(tmp_path, "--backbone", "alter", "--lasso-C", "0.01")
    with pytest.raises(ABIDECLIError, match="TempLasso arguments"):
        build_single_candidate(args, split_identity="sha256:split42")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--alter-layers", "3"),
        ("--alter-dropout", "0.2"),
        ("--alter-pos-encoding", "none"),
        ("--alter-pe-dim", "16"),
        ("--alter-heads", "2"),
        ("--alter-ffn-dim", "512"),
        ("--alter-cluster-num", "50"),
        ("--alter-edge-threshold", "0.3"),
        ("--alter-transformer-bias", "false"),
        ("--alter-variant", "harness"),
    ],
)
def test_bqn_backbone_rejects_every_alter_argument(tmp_path, flag, value) -> None:
    args = _args(tmp_path, *BQN_ARGS, flag, value)
    with pytest.raises(ABIDECLIError, match="must not define ALTER arguments"):
        build_single_candidate(args, split_identity="sha256:split42")


def test_templasso_still_rejects_alter_arguments(tmp_path) -> None:
    args = _args(
        tmp_path, *TEMPORAL_ARGS, "--lasso-C", "0.01", "--alter-layers", "3",
        mode="templasso",
    )
    with pytest.raises(ABIDECLIError, match="must not define ALTER arguments"):
        build_single_candidate(args, split_identity="sha256:split42")


def test_existing_bqn_mode_error_strings_are_preserved(tmp_path) -> None:
    with pytest.raises(ABIDECLIError, match="--bqn-layers"):
        build_single_candidate(
            _args(tmp_path), split_identity="sha256:split42"
        )
    with pytest.raises(ABIDECLIError, match="Temporal mode requires"):
        build_single_candidate(
            _args(tmp_path, *BQN_ARGS, mode="temporal"),
            split_identity="sha256:split42",
        )


# ------------------------------------------------------- orchestrator hand-off
def test_run_from_args_passes_an_alter_candidate_to_the_orchestrator(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    class FakeRuntimeFactory:
        split_identity = "sha256:split42"
        stored_split_seed = 42

        def __init__(self, config):
            captured["runtime_config"] = config

    def fake_search(candidates, *, search_dir, runtime_factory):
        captured["candidates"] = candidates
        captured["search_dir"] = search_dir
        return object()

    monkeypatch.setattr(cli_module, "ABIDERuntimeFactory", FakeRuntimeFactory)
    monkeypatch.setattr(cli_module, "run_validation_search", fake_search)

    run_from_args(_args(tmp_path, "--backbone", "alter"))

    assert len(captured["candidates"]) == 1
    identity = captured["candidates"][0].identity
    assert identity.mode == "vanilla"
    assert identity.backbone_config["name"] == "alter"
    assert captured["search_dir"] == tmp_path / "search"
