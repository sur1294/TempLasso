from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import unified_bn.training.abide_cli as cli_module
from unified_bn.training.abide_cli import (
    ABIDECLIError,
    build_parser,
    build_single_candidate,
    build_training_config,
    run_from_args,
)


def _base_args(tmp_path: Path, *, mode: str) -> Namespace:
    parser = build_parser()

    argv = [
        "--mode", mode,
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

    if mode == "temporal":
        argv += [
            "--bigru-hidden-dim", "32",
            "--bigru-layers", "1",
            "--bigru-pooling", "final",
        ]

    return parser.parse_args(argv)


def test_parser_exposes_no_test_evaluation_argument() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert all("test" not in option.lower() for option in option_strings)


def test_default_training_config_matches_current_atlas_defaults(
    tmp_path,
) -> None:
    args = _base_args(tmp_path, mode="vanilla")
    training = build_training_config(args)

    assert training.epochs == 200
    assert training.batch_size == 16
    assert training.optimizer.name == "adam"
    assert training.optimizer.lr == pytest.approx(1e-4)
    assert training.optimizer.weight_decay == pytest.approx(1e-4)
    assert training.lr_schedule.name == "cosine"
    assert training.lr_schedule.min_lr == pytest.approx(1e-5)
    assert training.checkpoint_selection.metric == "val_loss"
    assert training.checkpoint_selection.mode == "min"
    assert training.early_stopping.enabled is False


def test_vanilla_cli_builds_exactly_one_candidate(tmp_path) -> None:
    args = _base_args(tmp_path, mode="vanilla")

    candidate = build_single_candidate(
        args,
        split_identity="sha256:abc",
    )

    assert candidate.identity.mode == "vanilla"
    assert candidate.identity.seed == 42
    assert candidate.identity.split_identity == "sha256:abc"
    assert candidate.identity.backbone_config["name"] == "bqn"
    assert candidate.identity.backbone_config["n_layers"] == 3
    assert candidate.identity.backbone_config["activation"] == "elu"
    assert candidate.identity.backbone_config["dropout"] == pytest.approx(0.3)
    assert candidate.identity.temporal_config is None
    assert candidate.identity.window_policy is None


def test_temporal_cli_builds_exactly_one_native_bigru_candidate(
    tmp_path,
) -> None:
    args = _base_args(tmp_path, mode="temporal")

    candidate = build_single_candidate(
        args,
        split_identity="sha256:def",
    )

    assert candidate.identity.mode == "temporal"
    assert candidate.identity.temporal_config["name"] == "bigru"
    assert candidate.identity.temporal_config["hidden_dim"] == 32
    assert candidate.identity.temporal_config["num_layers"] == 1
    assert candidate.identity.temporal_config["dropout"] == pytest.approx(0.0)
    assert candidate.identity.temporal_config["pooling"] == "final"
    assert candidate.identity.window_policy.policy == "native"


def test_temporal_cli_requires_core_bigru_arguments(tmp_path) -> None:
    args = _base_args(tmp_path, mode="vanilla")
    args.mode = "temporal"

    with pytest.raises(ABIDECLIError, match="Temporal mode requires"):
        build_single_candidate(
            args,
            split_identity="sha256:def",
        )


def test_vanilla_cli_rejects_temporal_arguments(tmp_path) -> None:
    args = _base_args(tmp_path, mode="vanilla")
    args.bigru_hidden_dim = 64

    with pytest.raises(
        ABIDECLIError,
        match="must not define temporal-model arguments",
    ):
        build_single_candidate(
            args,
            split_identity="sha256:abc",
        )


def test_one_layer_nonzero_bigru_dropout_is_rejected_by_search_contract(
    tmp_path,
) -> None:
    args = _base_args(tmp_path, mode="temporal")
    args.bigru_dropout = 0.3

    with pytest.raises(ValueError, match="inactive"):
        build_single_candidate(
            args,
            split_identity="sha256:def",
        )


def test_cli_training_hyperparameters_are_candidate_config(tmp_path) -> None:
    args = _base_args(tmp_path, mode="vanilla")
    args.epochs = 7
    args.batch_size = 8
    args.lr = 3e-4
    args.weight_decay = 2e-4
    args.lr_schedule = "constant"
    args.min_lr = 0.0

    candidate = build_single_candidate(
        args,
        split_identity="sha256:abc",
    )

    assert candidate.training.epochs == 7
    assert candidate.training.batch_size == 8
    assert candidate.training.optimizer.lr == pytest.approx(3e-4)
    assert candidate.training.optimizer.weight_decay == pytest.approx(2e-4)
    assert candidate.training.lr_schedule.name == "constant"
    assert candidate.training.lr_schedule.min_lr == pytest.approx(0.0)


def test_run_from_args_uses_runtime_split_identity_and_orchestrator(
    tmp_path,
    monkeypatch,
) -> None:
    args = _base_args(tmp_path, mode="vanilla")
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
        selected_best_checkpoint_path = "/tmp/best.pt"

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
    assert candidate.identity.split_identity == "sha256:runtime"
    assert calls["search_dir"] == args.search_dir
    assert calls["runtime_factory"] is not None


def test_run_from_args_rejects_seed_mismatch_before_search(
    tmp_path,
    monkeypatch,
) -> None:
    args = _base_args(tmp_path, mode="vanilla")

    class FakeRuntimeFactory:
        def __init__(self, config):
            self.split_identity = "sha256:runtime"
            self.stored_split_seed = 1042

    monkeypatch.setattr(
        cli_module,
        "ABIDERuntimeFactory",
        FakeRuntimeFactory,
    )

    with pytest.raises(ABIDECLIError, match="does not match"):
        run_from_args(args)


def test_cli_is_single_candidate_entrypoint_not_grid_runner(tmp_path) -> None:
    args = _base_args(tmp_path, mode="temporal")

    candidate = build_single_candidate(
        args,
        split_identity="sha256:def",
    )

    assert not isinstance(candidate, (list, tuple))
