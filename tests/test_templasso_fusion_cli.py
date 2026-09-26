"""CLI coverage for ``--mode fusion_templasso`` (X + TempLasso)."""
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

BQN = ["--bqn-layers", "3", "--bqn-activation", "elu", "--bqn-dropout", "0.3"]
BIGRU = ["--bigru-hidden-dim", "32", "--bigru-layers", "1", "--bigru-pooling", "final"]
LASSO = ["--lasso-C", "0.01"]


def _argv(tmp_path: Path, mode: str, *groups) -> list[str]:
    argv = [
        "--mode", mode, "--seed", "42",
        "--split-path", str(tmp_path / "split.npz"),
        "--static-h5", str(tmp_path / "static.h5"),
        "--temporal-h5", str(tmp_path / "temporal.h5"),
        "--search-dir", str(tmp_path / "search"),
        "--device", "cpu",
    ]
    for g in groups:
        argv += g
    return argv


def _args(tmp_path: Path, mode: str, *groups) -> Namespace:
    return build_parser().parse_args(_argv(tmp_path, mode, *groups))


def test_parser_accepts_fusion_templasso_and_exposes_no_test_argument() -> None:
    parser = build_parser()
    assert "fusion_templasso" in next(a for a in parser._actions if a.dest == "mode").choices
    for action in parser._actions:
        for opt in action.option_strings:
            assert "test" not in opt.lower(), opt


def test_builds_exactly_one_heterogeneous_fusion_candidate(tmp_path) -> None:
    cand = build_single_candidate(_args(tmp_path, "fusion_templasso", BQN, BIGRU, LASSO),
                                  split_identity="sha256:x")
    ident = cand.identity
    assert ident.mode == "fusion"
    assert ident.backbone_config["name"] == "bqn"
    assert ident.backbone_config["n_layers"] == 3
    assert ident.temporal_backbone_config["name"] == "lasso_edges"
    assert ident.temporal_backbone_config["C"] == 0.01
    assert ident.temporal_backbone_config["n_rois"] == ident.backbone_config["n_rois"] == 100
    assert ident.temporal_config["name"] == "bigru" and ident.temporal_config["hidden_dim"] == 32
    assert ident.fusion_config == {"name": "concat"}
    assert ident.window_policy.policy == "native"


@pytest.mark.parametrize("missing,present,match", [
    ("bqn",   (BIGRU, LASSO), "requires the BQN backbone arguments"),
    ("bigru", (BQN, LASSO),   "Fusion-TempLasso mode requires"),
    ("lasso", (BQN, BIGRU),   "requires: --lasso-C"),
])
def test_requires_all_three_argument_groups(tmp_path, missing, present, match) -> None:
    with pytest.raises(ABIDECLIError, match=match):
        build_single_candidate(_args(tmp_path, "fusion_templasso", *present),
                               split_identity="sha256:x")


def test_plain_fusion_still_rejects_lasso_arguments(tmp_path) -> None:
    with pytest.raises(ABIDECLIError, match="must not define TempLasso arguments"):
        build_single_candidate(_args(tmp_path, "fusion", BQN, BIGRU, LASSO), split_identity="sha256:x")


def test_templasso_still_rejects_bqn_arguments(tmp_path) -> None:
    with pytest.raises(ABIDECLIError, match="no BQN backbone"):
        build_single_candidate(_args(tmp_path, "templasso", BQN, BIGRU, LASSO), split_identity="sha256:x")


def test_run_from_args_hands_fusion_candidate_to_orchestrator(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path, "fusion_templasso", BQN, BIGRU, LASSO)
    calls = {}

    class FakeRuntimeFactory:
        def __init__(self, config):
            self.split_identity = "sha256:runtime"; self.stored_split_seed = 42
        def __call__(self, candidate, index, run_dir):
            raise AssertionError("orchestrator is monkeypatched")

    class FakeSelection:
        selected_index = 0; selected_fingerprint = "a" * 64
        selection_metric = "val_loss"; selection_mode = "min"; selection_value = 0.5
        selected_best_checkpoint_path = "/dev/null/best.pt"

    def fake_search(candidates, *, search_dir, runtime_factory):
        calls["candidates"] = candidates; return FakeSelection()

    monkeypatch.setattr(cli_module, "ABIDERuntimeFactory", FakeRuntimeFactory)
    monkeypatch.setattr(cli_module, "run_validation_search", fake_search)

    run_from_args(args)
    cand = calls["candidates"][0]
    assert cand.identity.mode == "fusion"
    assert cand.identity.backbone_config["name"] == "bqn"
    assert cand.identity.temporal_backbone_config["name"] == "lasso_edges"
    assert cand.identity.split_identity == "sha256:runtime"
