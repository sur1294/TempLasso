"""
CLI coverage for ``--mode templasso``.

Mirrors the Stage-6J/6K-C5 CLI tests: argument contract, single-candidate
materialisation, fail-closed cross-mode argument checks, and the
``run_from_args`` -> orchestrator hand-off with the runtime and search
monkeypatched.

Also pins the one deliberate behaviour change this mode introduced: the BQN
architecture flags moved from argparse ``required=True`` to a per-mode check,
so vanilla/temporal/fusion without them now raise ``ABIDECLIError`` (the same
error family the BiGRU flags already used) instead of an argparse exit.
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


def _argv(tmp_path: Path, mode: str, *, bqn: bool, bigru: bool, lasso: bool):
    argv = [
        "--mode", mode,
        "--seed", "42",
        "--split-path", str(tmp_path / "split.npz"),
        "--static-h5", str(tmp_path / "static.h5"),
        "--temporal-h5", str(tmp_path / "temporal.h5"),
        "--search-dir", str(tmp_path / "search"),
        "--device", "cpu",
    ]
    if bqn:
        argv += ["--bqn-layers", "3", "--bqn-activation", "elu", "--bqn-dropout", "0.3"]
    if bigru:
        argv += ["--bigru-hidden-dim", "32", "--bigru-layers", "1", "--bigru-pooling", "final"]
    if lasso:
        argv += ["--lasso-C", "0.01"]
    return argv


def _args(tmp_path: Path, mode: str, **flags) -> Namespace:
    return build_parser().parse_args(_argv(tmp_path, mode, **flags))


def _templasso_args(tmp_path: Path, **extra) -> Namespace:
    argv = _argv(tmp_path, "templasso", bqn=False, bigru=True, lasso=True)
    for k, v in extra.items():
        argv += [k, str(v)]
    return build_parser().parse_args(argv)


# ------------------------------------------------------------- parser
def test_parser_accepts_templasso_and_exposes_no_test_argument() -> None:
    parser = build_parser()
    choices = next(a for a in parser._actions if a.dest == "mode").choices
    assert "templasso" in choices
    for action in parser._actions:
        for opt in action.option_strings:
            assert "test" not in opt.lower(), opt


# ------------------------------------------------- candidate materialisation
def test_templasso_cli_builds_exactly_one_lasso_edges_temporal_candidate(tmp_path) -> None:
    args = _templasso_args(tmp_path)
    cand = build_single_candidate(args, split_identity="sha256:x")

    ident = cand.identity
    assert ident.mode == "temporal"                      # reuses the generic temporal path
    assert ident.backbone_config["name"] == "lasso_edges"
    assert ident.backbone_config["C"] == 0.01
    assert ident.backbone_config["input_space"] == "fisher_z"
    assert ident.backbone_config["n_rois"] == 100
    assert ident.temporal_config["name"] == "bigru"
    assert ident.temporal_config["hidden_dim"] == 32
    assert ident.window_policy.policy == "native"
    assert ident.fusion_config is None
    assert cand.training.epochs == 200 and cand.training.batch_size == 16


def test_templasso_cli_honours_input_space_and_expected_n_rois(tmp_path) -> None:
    args = _templasso_args(tmp_path, **{"--lasso-input-space": "correlation", "--expected-n-rois": 6})
    cand = build_single_candidate(args, split_identity="sha256:x")
    assert cand.identity.backbone_config["input_space"] == "correlation"
    assert cand.identity.backbone_config["n_rois"] == 6


# ------------------------------------------------------ fail-closed checks
def test_templasso_cli_requires_lasso_C(tmp_path) -> None:
    args = _args(tmp_path, "templasso", bqn=False, bigru=True, lasso=False)
    with pytest.raises(ABIDECLIError, match="requires: --lasso-C"):
        build_single_candidate(args, split_identity="sha256:x")


def test_templasso_cli_requires_core_bigru_arguments(tmp_path) -> None:
    args = _args(tmp_path, "templasso", bqn=False, bigru=False, lasso=True)
    with pytest.raises(ABIDECLIError, match="TempLasso mode requires"):
        build_single_candidate(args, split_identity="sha256:x")


def test_templasso_cli_rejects_bqn_arguments(tmp_path) -> None:
    args = _args(tmp_path, "templasso", bqn=True, bigru=True, lasso=True)
    with pytest.raises(ABIDECLIError, match="no BQN backbone"):
        build_single_candidate(args, split_identity="sha256:x")


@pytest.mark.parametrize("mode", ["vanilla", "temporal", "fusion"])
def test_bqn_modes_still_require_bqn_arguments_as_cli_error(tmp_path, mode) -> None:
    # Behaviour change pinned: missing BQN flags are now an ABIDECLIError
    # from build_single_candidate, matching how missing BiGRU flags behave.
    args = _args(tmp_path, mode, bqn=False, bigru=(mode != "vanilla"), lasso=False)
    with pytest.raises(ABIDECLIError, match="requires the BQN backbone arguments"):
        build_single_candidate(args, split_identity="sha256:x")


@pytest.mark.parametrize("mode", ["vanilla", "temporal", "fusion"])
def test_bqn_modes_reject_lasso_arguments(tmp_path, mode) -> None:
    args = _args(tmp_path, mode, bqn=True, bigru=(mode != "vanilla"), lasso=True)
    with pytest.raises(ABIDECLIError, match="must not define TempLasso arguments"):
        build_single_candidate(args, split_identity="sha256:x")


def test_existing_temporal_and_fusion_error_strings_are_preserved(tmp_path) -> None:
    # The Stage-6J/6K-C5 tests match these exact prefixes; the mode-name map
    # must not have changed them.
    with pytest.raises(ABIDECLIError, match="Temporal mode requires"):
        build_single_candidate(_args(tmp_path, "temporal", bqn=True, bigru=False, lasso=False),
                               split_identity="sha256:x")
    with pytest.raises(ABIDECLIError, match="Fusion mode requires"):
        build_single_candidate(_args(tmp_path, "fusion", bqn=True, bigru=False, lasso=False),
                               split_identity="sha256:x")


# -------------------------------------------------- run_from_args hand-off
def test_templasso_run_from_args_reaches_orchestrator_with_lasso_candidate(
    tmp_path, monkeypatch
) -> None:
    args = _templasso_args(tmp_path)
    calls = {}

    class FakeRuntimeFactory:
        def __init__(self, config):
            calls["runtime_config"] = config
            self.split_identity = "sha256:runtime"
            self.stored_split_seed = 42

        def __call__(self, candidate, index, run_dir):
            raise AssertionError("orchestrator is monkeypatched in this test")

    class FakeSelection:
        selected_index = 0
        selected_fingerprint = "a" * 64
        selection_metric = "val_loss"
        selection_mode = "min"
        selection_value = 0.5
        selected_best_checkpoint_path = "/dev/null/best.pt"

    def fake_search(candidates, *, search_dir, runtime_factory):
        calls["candidates"] = candidates
        calls["search_dir"] = search_dir
        return FakeSelection()

    monkeypatch.setattr(cli_module, "ABIDERuntimeFactory", FakeRuntimeFactory)
    monkeypatch.setattr(cli_module, "run_validation_search", fake_search)

    selection = run_from_args(args)

    assert selection.selected_index == 0
    assert len(calls["candidates"]) == 1
    cand = calls["candidates"][0]
    assert cand.identity.mode == "temporal"
    assert cand.identity.backbone_config["name"] == "lasso_edges"
    assert cand.identity.split_identity == "sha256:runtime"
    assert calls["search_dir"] == args.search_dir
