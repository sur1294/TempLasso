from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import unified_bn.training.orchestrator as orchestration_module
from unified_bn.training.config import (
    TrainingConfig,
    ValidationSelectionRule,
)
from unified_bn.training.metrics import BinaryClassificationMetrics
from unified_bn.training.orchestrator import (
    CandidateRuntime,
    SearchOrchestrationError,
    candidate_run_directory,
    run_validation_search,
)
from unified_bn.training.results import (
    CandidateTrainingResult,
    write_candidate_result,
)
from unified_bn.training.search_space import (
    BQNSearchSpace,
    candidate_fingerprint,
    generate_vanilla_bqn_candidates,
)


def _candidates():
    return generate_vanilla_bqn_candidates(
        seed=42,
        split_identity="sha256:split42",
        backbone_space=BQNSearchSpace(
            n_layers=(1, 2, 3),
            activations=("elu",),
            dropouts=(0.0,),
        ),
    )


def _metrics(loss: float) -> BinaryClassificationMetrics:
    return BinaryClassificationMetrics(
        loss=loss,
        roc_auc=0.7,
        accuracy=0.6,
        sensitivity=0.5,
        specificity=0.7,
        n_samples=20,
    )


def _fake_result(candidate, run_dir: Path, value: float):
    run_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "best.pt"
    last = run_dir / "last.pt"
    history = run_dir / "history.json"
    result_path = run_dir / "result.json"

    best.write_bytes(b"best")
    last.write_bytes(b"last")
    history.write_text("{}\n", encoding="utf-8")

    metrics = _metrics(value)
    result = CandidateTrainingResult(
        run_config=candidate.to_dict(),
        requested_epochs=200,
        completed_epochs=200,
        stopped_early=False,
        best_epoch=10,
        selection_metric=candidate.training.checkpoint_selection.metric,
        selection_mode=candidate.training.checkpoint_selection.mode,
        best_selection_value=value,
        best_validation_metrics=metrics,
        restored_validation_metrics=metrics,
        best_checkpoint_path=str(best),
        last_checkpoint_path=str(last),
        history_path=str(history),
        result_path=str(result_path),
        loss_function="torch.nn.modules.loss.CrossEntropyLoss",
        runtime_provenance={"test": "fake"},
        user_provenance={},
        test_metrics=None,
    )
    write_candidate_result(result)
    return result


def _runtime():
    return CandidateRuntime(
        model=nn.Linear(2, 2),
        train_batches=(),
        val_batches=(),
        device="cpu",
    )


def test_candidate_run_directory_is_deterministic_and_fingerprinted(
    tmp_path,
) -> None:
    candidate = _candidates()[1]

    first = candidate_run_directory(
        tmp_path,
        index=1,
        candidate=candidate,
    )
    second = candidate_run_directory(
        tmp_path,
        index=1,
        candidate=candidate,
    )

    assert first == second
    assert first.name.startswith("candidate_0001_")
    assert candidate_fingerprint(candidate)[:12] in first.name


def test_orchestrator_calls_existing_trainer_once_per_candidate_and_selects(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _candidates()
    calls = []

    values = {
        candidate_fingerprint(candidates[0]): 0.55,
        candidate_fingerprint(candidates[1]): 0.40,
        candidate_fingerprint(candidates[2]): 0.48,
    }

    def runtime_factory(candidate, index, run_dir):
        calls.append(("factory", index, candidate_fingerprint(candidate)))
        return _runtime()

    def fake_train(
        model,
        train_batches,
        val_batches,
        *,
        run_config,
        run_dir,
        device,
        loss_fn=None,
        dataloader_generator=None,
        resume_from=None,
        user_provenance=None,
        non_blocking=False,
    ):
        calls.append(
            (
                "train",
                candidate_fingerprint(run_config),
                None if resume_from is None else Path(resume_from).name,
            )
        )
        return _fake_result(
            run_config,
            Path(run_dir),
            values[candidate_fingerprint(run_config)],
        )

    monkeypatch.setattr(
        orchestration_module,
        "train_candidate",
        fake_train,
    )

    selection = run_validation_search(
        candidates,
        search_dir=tmp_path / "search",
        runtime_factory=runtime_factory,
    )

    assert selection.selected_index == 1
    assert selection.selection_value == pytest.approx(0.40)

    factory_calls = [item for item in calls if item[0] == "factory"]
    train_calls = [item for item in calls if item[0] == "train"]
    assert len(factory_calls) == 3
    assert len(train_calls) == 3
    assert all(item[2] is None for item in train_calls)


def test_completed_validation_candidates_are_reused_without_runtime_factory(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _candidates()
    root = tmp_path / "search"

    values = (0.55, 0.40, 0.48)
    for index, (candidate, value) in enumerate(zip(candidates, values)):
        run_dir = candidate_run_directory(
            root,
            index=index,
            candidate=candidate,
        )
        _fake_result(candidate, run_dir, value)

    def forbidden_runtime(*args, **kwargs):
        raise AssertionError(
            "runtime_factory must not run for completed candidates"
        )

    def forbidden_train(*args, **kwargs):
        raise AssertionError(
            "train_candidate must not run for completed candidates"
        )

    monkeypatch.setattr(
        orchestration_module,
        "train_candidate",
        forbidden_train,
    )

    selection = run_validation_search(
        candidates,
        search_dir=root,
        runtime_factory=forbidden_runtime,
    )

    assert selection.selected_index == 1
    assert (root / "selection.json").is_file()


def test_interrupted_candidate_resumes_from_last_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _candidates()[0]
    root = tmp_path / "search"
    run_dir = candidate_run_directory(
        root,
        index=0,
        candidate=candidate,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "last.pt").write_bytes(b"resume")
    (run_dir / "history.json").write_text("{}\n", encoding="utf-8")

    seen = {}

    def fake_train(
        model,
        train_batches,
        val_batches,
        *,
        run_config,
        run_dir,
        device,
        loss_fn=None,
        dataloader_generator=None,
        resume_from=None,
        user_provenance=None,
        non_blocking=False,
    ):
        seen["resume_from"] = Path(resume_from)
        return _fake_result(run_config, Path(run_dir), 0.50)

    monkeypatch.setattr(
        orchestration_module,
        "train_candidate",
        fake_train,
    )

    selection = run_validation_search(
        (candidate,),
        search_dir=root,
        runtime_factory=lambda *args: _runtime(),
    )

    assert seen["resume_from"] == run_dir / "last.pt"
    assert selection.selected_index == 0


def test_unsafe_partial_artifact_state_fails_closed(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _candidates()[0]
    root = tmp_path / "search"
    run_dir = candidate_run_directory(
        root,
        index=0,
        candidate=candidate,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text("{}\n", encoding="utf-8")

    def forbidden_runtime(*args, **kwargs):
        raise AssertionError("runtime_factory must not run")

    with pytest.raises(SearchOrchestrationError, match="unsafe partial"):
        run_validation_search(
            (candidate,),
            search_dir=root,
            runtime_factory=forbidden_runtime,
        )


def test_seed_is_applied_before_runtime_model_construction(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _candidates()[:2]
    initial_weights = []

    def runtime_factory(candidate, index, run_dir):
        model = nn.Linear(4, 2)
        initial_weights.append(model.weight.detach().clone())
        return CandidateRuntime(
            model=model,
            train_batches=(),
            val_batches=(),
            device="cpu",
        )

    def fake_train(
        model,
        train_batches,
        val_batches,
        *,
        run_config,
        run_dir,
        device,
        loss_fn=None,
        dataloader_generator=None,
        resume_from=None,
        user_provenance=None,
        non_blocking=False,
    ):
        return _fake_result(
            run_config,
            Path(run_dir),
            0.5 + 0.01 * run_config.identity.backbone_config["n_layers"],
        )

    monkeypatch.setattr(
        orchestration_module,
        "train_candidate",
        fake_train,
    )

    run_validation_search(
        candidates,
        search_dir=tmp_path / "search",
        runtime_factory=runtime_factory,
    )

    assert len(initial_weights) == 2
    assert torch.equal(initial_weights[0], initial_weights[1])


def test_search_preflight_rejects_mixed_seeds_before_any_runtime_call(
    tmp_path,
) -> None:
    base = _candidates()[0]
    other = replace(
        base,
        identity=replace(base.identity, seed=1042),
    )

    calls = []

    def runtime_factory(*args):
        calls.append(args)
        return _runtime()

    with pytest.raises(SearchOrchestrationError, match="random seeds"):
        run_validation_search(
            (base, other),
            search_dir=tmp_path / "search",
            runtime_factory=runtime_factory,
        )

    assert calls == []


def test_search_preflight_rejects_mixed_split_identity(
    tmp_path,
) -> None:
    base = _candidates()[0]
    other = replace(
        base,
        identity=replace(
            base.identity,
            split_identity="sha256:different",
        ),
    )

    with pytest.raises(SearchOrchestrationError, match="split identities"):
        run_validation_search(
            (base, other),
            search_dir=tmp_path / "search",
            runtime_factory=lambda *args: _runtime(),
        )


def test_search_preflight_rejects_mixed_selection_rules(
    tmp_path,
) -> None:
    base = _candidates()[0]
    changed_training = TrainingConfig(
        checkpoint_selection=ValidationSelectionRule(
            metric="val_auc",
            mode="max",
        )
    )
    other = replace(base, training=changed_training)

    with pytest.raises(
        SearchOrchestrationError,
        match="selection rules",
    ):
        run_validation_search(
            (base, other),
            search_dir=tmp_path / "search",
            runtime_factory=lambda *args: _runtime(),
        )


def test_completed_result_with_test_metrics_is_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _candidates()[0]
    root = tmp_path / "search"
    run_dir = candidate_run_directory(
        root,
        index=0,
        candidate=candidate,
    )
    result = _fake_result(candidate, run_dir, 0.5)

    payload = result.to_dict()
    payload["status"] = "final_test_complete"
    payload["test_metrics"] = _metrics(0.4).to_dict()
    (run_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SearchOrchestrationError,
        match="validation-complete",
    ):
        run_validation_search(
            (candidate,),
            search_dir=root,
            runtime_factory=lambda *args: _runtime(),
        )


def test_existing_selection_must_reproduce_exactly(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _candidates()
    root = tmp_path / "search"

    values = (0.55, 0.40, 0.48)
    for index, (candidate, value) in enumerate(zip(candidates, values)):
        run_dir = candidate_run_directory(
            root,
            index=index,
            candidate=candidate,
        )
        _fake_result(candidate, run_dir, value)

    first = run_validation_search(
        candidates,
        search_dir=root,
        runtime_factory=lambda *args: _runtime(),
    )
    assert first.selected_index == 1

    payload = json.loads(
        (root / "selection.json").read_text(encoding="utf-8")
    )
    payload["selected_index"] = 0
    (root / "selection.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SearchOrchestrationError,
        match="disagrees",
    ):
        run_validation_search(
            candidates,
            search_dir=root,
            runtime_factory=lambda *args: _runtime(),
        )


def test_search_does_not_expose_any_test_loader_argument() -> None:
    import inspect

    signature = inspect.signature(run_validation_search)

    assert "test_batches" not in signature.parameters
    assert "test_loader" not in signature.parameters
    assert "test_data" not in signature.parameters
