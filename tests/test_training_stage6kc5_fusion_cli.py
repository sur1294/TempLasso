from __future__ import annotations

from pathlib import Path

import pytest

import unified_bn.training.abide_cli as cli_module
from unified_bn.training.abide_cli import (
    ABIDECLIError,
    build_parser,
    build_single_candidate,
    run_from_args,
)


def _fusion_args(tmp_path: Path):
    parser = build_parser()
    return parser.parse_args(
        [
            "--mode", "fusion",
            "--seed", "42",
            "--split-path", str(tmp_path / "split.npz"),
            "--static-h5", str(tmp_path / "static.h5"),
            "--temporal-h5", str(tmp_path / "temporal.h5"),
            "--search-dir", str(tmp_path / "search"),
            "--device", "cpu",
            "--bqn-layers", "3",
            "--bqn-activation", "elu",
            "--bqn-dropout", "0.3",
            "--bigru-hidden-dim", "32",
            "--bigru-layers", "1",
            "--bigru-pooling", "final",
        ]
    )


def test_parser_accepts_fusion_but_still_exposes_no_test_argument(tmp_path):
    args = _fusion_args(tmp_path)
    assert args.mode == "fusion"

    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert all("test" not in option.lower() for option in option_strings)


def test_fusion_cli_builds_exactly_one_native_concat_candidate(tmp_path):
    args = _fusion_args(tmp_path)

    candidate = build_single_candidate(
        args,
        split_identity="sha256:fusion",
    )

    assert candidate.identity.mode == "fusion"
    assert candidate.identity.seed == 42
    assert candidate.identity.split_identity == "sha256:fusion"
    assert candidate.identity.backbone_config["name"] == "bqn"
    assert candidate.identity.backbone_config["n_layers"] == 3
    assert candidate.identity.backbone_config["activation"] == "elu"
    assert candidate.identity.backbone_config["dropout"] == pytest.approx(0.3)
    assert candidate.identity.temporal_config["name"] == "bigru"
    assert candidate.identity.temporal_config["hidden_dim"] == 32
    assert candidate.identity.temporal_config["num_layers"] == 1
    assert candidate.identity.temporal_config["dropout"] == pytest.approx(0.0)
    assert candidate.identity.temporal_config["pooling"] == "final"
    assert candidate.identity.window_policy.policy == "native"
    assert candidate.identity.fusion_config == {"name": "concat"}


def test_fusion_cli_requires_core_bigru_arguments(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--mode", "fusion",
            "--seed", "42",
            "--split-path", str(tmp_path / "split.npz"),
            "--static-h5", str(tmp_path / "static.h5"),
            "--temporal-h5", str(tmp_path / "temporal.h5"),
            "--search-dir", str(tmp_path / "search"),
            "--device", "cpu",
            "--bqn-layers", "3",
            "--bqn-activation", "elu",
            "--bqn-dropout", "0.3",
        ]
    )

    with pytest.raises(ABIDECLIError, match="Fusion mode requires"):
        build_single_candidate(
            args,
            split_identity="sha256:fusion",
        )


def test_fusion_one_layer_nonzero_bigru_dropout_is_rejected(tmp_path):
    args = _fusion_args(tmp_path)
    args.bigru_dropout = 0.2

    with pytest.raises(ValueError, match="inactive"):
        build_single_candidate(
            args,
            split_identity="sha256:fusion",
        )


def test_fusion_run_from_args_uses_runtime_identity_and_orchestrator(
    tmp_path,
    monkeypatch,
):
    args = _fusion_args(tmp_path)
    calls = {}

    class FakeRuntimeFactory:
        def __init__(self, config):
            calls["runtime_config"] = config
            self.split_identity = "sha256:runtime-fusion"
            self.stored_split_seed = 42

        def __call__(self, candidate, index, run_dir):
            raise AssertionError("orchestrator is monkeypatched in this test")

    class FakeSelection:
        selected_index = 0
        selected_fingerprint = "f" * 64
        selection_metric = "val_loss"
        selection_mode = "min"
        selection_value = 0.4
        selected_best_checkpoint_path = "/tmp/fusion_best.pt"

    def fake_search(candidates, *, search_dir, runtime_factory):
        calls["candidates"] = candidates
        calls["search_dir"] = search_dir
        calls["runtime_factory"] = runtime_factory
        return FakeSelection()

    monkeypatch.setattr(
        cli_module,
        "ABIDERuntimeFactory",
        FakeRuntimeFactory,
    )
    monkeypatch.setattr(
        cli_module,
        "run_validation_search",
        fake_search,
    )

    selection = run_from_args(args)

    assert selection.selected_index == 0
    assert len(calls["candidates"]) == 1
    candidate = calls["candidates"][0]
    assert candidate.identity.mode == "fusion"
    assert candidate.identity.split_identity == "sha256:runtime-fusion"
    assert candidate.identity.fusion_config == {"name": "concat"}
    assert calls["search_dir"] == args.search_dir
    assert calls["runtime_factory"] is not None
