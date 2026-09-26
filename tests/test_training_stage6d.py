from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from unified_bn.data.collate import StaticBatch
from unified_bn.training.config import (
    CandidateRunConfig,
    EarlyStoppingConfig,
    LRScheduleConfig,
    OptimizerConfig,
    RunIdentity,
    TrainingConfig,
    ValidationSelectionRule,
)
from unified_bn.training.metrics import BinaryClassificationMetrics
from unified_bn.training.optim import (
    EpochLRScheduler,
    OptimizationConfigError,
    build_lr_scheduler,
    build_optimizer,
)
from unified_bn.training.results import ResultArtifactError, read_history_json
from unified_bn.training.trainer import (
    TrainerError,
    evaluate_final_test_and_record,
    train_candidate,
)
import unified_bn.training.trainer as trainer_module


@dataclass(frozen=True)
class _Output:
    logits: torch.Tensor


class _ScriptedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0))

    def forward(self, fc: torch.Tensor) -> _Output:
        score = self.marker.expand(fc.shape[0])
        return _Output(logits=torch.stack([-score, score], dim=1))


class _TinyStaticPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, fc: torch.Tensor) -> _Output:
        x = fc[:, 0, 0].unsqueeze(1)
        return _Output(logits=self.linear(x))


class _CountingBatches:
    def __init__(self, batches):
        self.batches = list(batches)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.batches)


def _batch() -> StaticBatch:
    return StaticBatch(
        subject_ids=("s0", "s1", "s2", "s3"),
        labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        fc=torch.tensor([[[0.0]], [[1.0]], [[0.2]], [[0.8]]], dtype=torch.float32),
        source_indices=torch.arange(4, dtype=torch.long),
    )


def _metrics(loss: float, auc: float = 0.7) -> BinaryClassificationMetrics:
    return BinaryClassificationMetrics(
        loss=float(loss),
        roc_auc=float(auc),
        accuracy=0.75,
        sensitivity=0.5,
        specificity=1.0,
        n_samples=4,
    )


def _run_config(
    *,
    epochs: int = 3,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    min_lr: float = 1e-5,
    schedule: str = "cosine",
    early_stopping: EarlyStoppingConfig | None = None,
) -> CandidateRunConfig:
    training = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        optimizer=OptimizerConfig(
            name="adam",
            lr=lr,
            weight_decay=weight_decay,
        ),
        lr_schedule=LRScheduleConfig(
            name=schedule,
            min_lr=min_lr,
        ),
        checkpoint_selection=ValidationSelectionRule(
            metric="val_loss",
            mode="min",
        ),
        early_stopping=(
            EarlyStoppingConfig()
            if early_stopping is None
            else early_stopping
        ),
    )
    return CandidateRunConfig(
        identity=RunIdentity(
            mode="vanilla",
            seed=42,
            split_identity="sha256:test-split",
            backbone_config={"name": "dummy", "depth": 3},
        ),
        training=training,
    )


def _install_scripted_epoch_runner(monkeypatch, *, val_losses, val_aucs=None):
    val_aucs = list(val_aucs or [0.7] * len(val_losses))
    val_losses = list(val_losses)
    state = {"train_calls": 0, "val_calls": 0}

    def fake_run_binary_epoch(
        model,
        batches,
        *,
        loss_fn,
        device,
        training,
        optimizer=None,
        non_blocking=False,
    ):
        if training:
            state["train_calls"] += 1
            with torch.no_grad():
                model.marker.fill_(float(state["train_calls"]))
            return _metrics(1.0 - 0.05 * state["train_calls"], auc=0.55)

        state["val_calls"] += 1
        # The extra validation call after best-checkpoint restoration should
        # report the metrics belonging to the restored marker/best epoch.
        if state["val_calls"] <= len(val_losses):
            index = state["val_calls"] - 1
        else:
            index = int(round(float(model.marker.detach().cpu().item()))) - 1
        return _metrics(val_losses[index], auc=val_aucs[index])

    monkeypatch.setattr(trainer_module, "run_binary_epoch", fake_run_binary_epoch)
    return state


def test_optimizer_factory_uses_candidate_lr_and_weight_decay_not_constants():
    model = nn.Linear(2, 2)
    config = OptimizerConfig(name="adam", lr=3e-4, weight_decay=7e-5)
    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(7e-5)


def test_optimizer_factory_rejects_model_without_trainable_parameters():
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(OptimizationConfigError, match="no trainable parameters"):
        build_optimizer(model, OptimizerConfig())


def test_cosine_scheduler_reads_arbitrary_base_floor_and_horizon_from_config():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3, weight_decay=0.0)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule_config=LRScheduleConfig(name="cosine", min_lr=0.05),
        total_epochs=5,
    )

    observed = [scheduler.set_epoch(i)[0] for i in range(5)]
    assert observed[0] == pytest.approx(0.3)
    assert observed[-1] == pytest.approx(0.05)
    assert observed[1] < observed[0]
    assert observed[2] < observed[1]
    assert observed[3] < observed[2]


def test_constant_scheduler_preserves_configured_base_lr():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = EpochLRScheduler(
        optimizer,
        schedule_config=LRScheduleConfig(name="constant", min_lr=0.0),
        total_epochs=4,
    )
    assert [scheduler.set_epoch(i)[0] for i in range(4)] == pytest.approx([2e-4] * 4)


def test_scheduler_state_roundtrip_restores_epoch_and_lr():
    model1 = nn.Linear(1, 1)
    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    sch1 = EpochLRScheduler(
        opt1,
        schedule_config=LRScheduleConfig(name="cosine", min_lr=1e-5),
        total_epochs=7,
    )
    sch1.set_epoch(3)

    model2 = nn.Linear(1, 1)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    sch2 = EpochLRScheduler(
        opt2,
        schedule_config=LRScheduleConfig(name="cosine", min_lr=1e-5),
        total_epochs=7,
    )
    sch2.load_state_dict(sch1.state_dict())

    assert sch2.last_epoch_index == 3
    assert sch2.get_last_lr() == pytest.approx(sch1.get_last_lr())


def test_train_candidate_selects_minimum_validation_loss_not_maximum_auc(
    tmp_path: Path, monkeypatch
):
    state = _install_scripted_epoch_runner(
        monkeypatch,
        val_losses=[0.50, 0.30, 0.40],
        val_aucs=[0.60, 0.70, 0.99],
    )
    model = _ScriptedModel()
    config = _run_config(epochs=3)

    result = train_candidate(
        model,
        [_batch()],
        [_batch()],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
    )

    assert state["train_calls"] == 3
    assert result.completed_epochs == 3
    assert result.stopped_early is False
    assert result.best_epoch == 2
    assert result.best_selection_value == pytest.approx(0.30)
    assert result.best_validation_metrics.roc_auc == pytest.approx(0.70)
    # AUC was highest at epoch 3, but ABIDE selection is minimum val loss.
    assert float(model.marker.detach()) == pytest.approx(2.0)


def test_atlas_default_early_stopping_off_runs_full_configured_budget(
    tmp_path: Path, monkeypatch
):
    state = _install_scripted_epoch_runner(
        monkeypatch,
        val_losses=[0.5, 0.6, 0.7, 0.8],
    )
    config = _run_config(epochs=4)

    result = train_candidate(
        _ScriptedModel(),
        [_batch()],
        [_batch()],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
    )

    assert state["train_calls"] == 4
    assert result.completed_epochs == 4
    assert result.stopped_early is False


def test_optional_early_stopping_is_separate_from_checkpoint_selection(
    tmp_path: Path, monkeypatch
):
    state = _install_scripted_epoch_runner(
        monkeypatch,
        val_losses=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    config = _run_config(
        epochs=5,
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=2,
            min_delta=0.0,
            rule=ValidationSelectionRule(metric="val_loss", mode="min"),
        ),
    )

    result = train_candidate(
        _ScriptedModel(),
        [_batch()],
        [_batch()],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
    )

    assert state["train_calls"] == 3
    assert result.completed_epochs == 3
    assert result.stopped_early is True
    assert result.best_epoch == 1


def test_best_checkpoint_restored_and_revalidated_before_return(
    tmp_path: Path, monkeypatch
):
    _install_scripted_epoch_runner(
        monkeypatch,
        val_losses=[0.4, 0.2, 0.35],
        val_aucs=[0.61, 0.72, 0.80],
    )
    model = _ScriptedModel()

    result = train_candidate(
        model,
        [_batch()],
        [_batch()],
        run_config=_run_config(epochs=3),
        run_dir=tmp_path,
        device="cpu",
    )

    assert result.best_epoch == 2
    assert float(model.marker.detach()) == pytest.approx(2.0)
    assert result.restored_validation_metrics.to_dict() == result.best_validation_metrics.to_dict()


def test_candidate_history_is_contiguous_and_contains_configured_learning_rates(
    tmp_path: Path, monkeypatch
):
    _install_scripted_epoch_runner(monkeypatch, val_losses=[0.4, 0.3, 0.35])
    config = _run_config(epochs=3, lr=4e-4, min_lr=2e-5)

    train_candidate(
        _ScriptedModel(),
        [_batch()],
        [_batch()],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
    )
    records = read_history_json(
        tmp_path / "history.json",
        expected_run_config=config,
    )

    assert [record.epoch for record in records] == [1, 2, 3]
    assert records[0].learning_rates[0] == pytest.approx(4e-4)
    assert records[-1].learning_rates[0] == pytest.approx(2e-5)


def test_tuning_candidate_result_contains_full_optimization_config_and_no_test(
    tmp_path: Path, monkeypatch
):
    _install_scripted_epoch_runner(monkeypatch, val_losses=[0.3, 0.2])
    config = _run_config(
        epochs=2,
        lr=3e-4,
        weight_decay=8e-5,
        min_lr=7e-6,
    )

    result = train_candidate(
        _ScriptedModel(),
        [_batch()],
        [_batch()],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
    )
    payload = json.loads((tmp_path / "result.json").read_text())

    assert result.test_metrics is None
    assert payload["test_metrics"] is None
    assert payload["status"] == "validation_complete"
    training = payload["run_config"]["training"]
    assert training["epochs"] == 2
    assert training["optimizer"]["lr"] == pytest.approx(3e-4)
    assert training["optimizer"]["weight_decay"] == pytest.approx(8e-5)
    assert training["lr_schedule"]["min_lr"] == pytest.approx(7e-6)


def test_train_candidate_api_has_no_test_loader_argument():
    parameters = inspect.signature(train_candidate).parameters
    assert "test_loader" not in parameters
    assert "test_batches" not in parameters
    assert all("test" not in name.lower() for name in parameters)


def test_test_data_access_occurs_only_through_explicit_final_test_call(tmp_path: Path):
    torch.manual_seed(5)
    model = _TinyStaticPipeline()
    config = _run_config(epochs=1)
    batch = _batch()

    result = train_candidate(
        model,
        [batch],
        [batch],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
    )

    test_batches = _CountingBatches([batch])
    assert test_batches.iterations == 0
    metrics = evaluate_final_test_and_record(
        model,
        test_batches,
        result_path=result.result_path,
        device="cpu",
    )
    assert test_batches.iterations == 1
    assert metrics.n_samples == 4

    payload = json.loads(Path(result.result_path).read_text())
    assert payload["status"] == "final_test_complete"
    assert payload["test_metrics"]["n_samples"] == 4

    with pytest.raises(ResultArtifactError, match="already recorded"):
        evaluate_final_test_and_record(
            model,
            test_batches,
            result_path=result.result_path,
            device="cpu",
        )


def test_fresh_run_refuses_to_overwrite_existing_artifacts(tmp_path: Path, monkeypatch):
    _install_scripted_epoch_runner(monkeypatch, val_losses=[0.3])
    (tmp_path / "history.json").write_text("do not overwrite")

    with pytest.raises(TrainerError, match="refuses to overwrite"):
        train_candidate(
            _ScriptedModel(),
            [_batch()],
            [_batch()],
            run_config=_run_config(epochs=1),
            run_dir=tmp_path,
            device="cpu",
        )

    assert (tmp_path / "history.json").read_text() == "do not overwrite"


def test_multi_epoch_candidate_rejects_one_shot_iterators(tmp_path: Path):
    one_shot = iter([_batch()])
    with pytest.raises(TrainerError, match="one-shot iterator"):
        train_candidate(
            _ScriptedModel(),
            one_shot,
            [_batch()],
            run_config=_run_config(epochs=2),
            run_dir=tmp_path,
            device="cpu",
        )


def test_resume_continues_from_last_checkpoint_without_repeating_completed_epochs(
    tmp_path: Path, monkeypatch
):
    config = _run_config(epochs=4)
    model1 = _ScriptedModel()
    state = {"train_calls": 0}
    losses = [0.5, 0.4, 0.3, 0.35]

    def interrupted_runner(
        model,
        batches,
        *,
        loss_fn,
        device,
        training,
        optimizer=None,
        non_blocking=False,
    ):
        if training:
            state["train_calls"] += 1
            if state["train_calls"] == 3:
                raise RuntimeError("simulated walltime interruption")
            with torch.no_grad():
                model.marker.fill_(float(state["train_calls"]))
            return _metrics(0.9)
        epoch = int(round(float(model.marker.detach())))
        return _metrics(losses[epoch - 1])

    monkeypatch.setattr(trainer_module, "run_binary_epoch", interrupted_runner)
    with pytest.raises(RuntimeError, match="simulated walltime"):
        train_candidate(
            model1,
            [_batch()],
            [_batch()],
            run_config=config,
            run_dir=tmp_path,
            device="cpu",
        )

    assert (tmp_path / "last.pt").is_file()
    history_before = read_history_json(
        tmp_path / "history.json", expected_run_config=config
    )
    assert [record.epoch for record in history_before] == [1, 2]

    resumed_training_epochs = []

    def resumed_runner(
        model,
        batches,
        *,
        loss_fn,
        device,
        training,
        optimizer=None,
        non_blocking=False,
    ):
        if training:
            next_epoch = int(round(float(model.marker.detach()))) + 1
            resumed_training_epochs.append(next_epoch)
            with torch.no_grad():
                model.marker.fill_(float(next_epoch))
            return _metrics(0.8)
        epoch = int(round(float(model.marker.detach())))
        return _metrics(losses[epoch - 1])

    monkeypatch.setattr(trainer_module, "run_binary_epoch", resumed_runner)
    model2 = _ScriptedModel()
    result = train_candidate(
        model2,
        [_batch()],
        [_batch()],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
        resume_from=tmp_path / "last.pt",
    )

    assert resumed_training_epochs == [3, 4]
    assert result.completed_epochs == 4
    assert result.best_epoch == 3
    assert float(model2.marker.detach()) == pytest.approx(3.0)
    history_after = read_history_json(
        tmp_path / "history.json", expected_run_config=config
    )
    assert [record.epoch for record in history_after] == [1, 2, 3, 4]


def test_resume_rejects_incompatible_optimization_configuration(tmp_path: Path, monkeypatch):
    _install_scripted_epoch_runner(monkeypatch, val_losses=[0.4, 0.3])
    config = _run_config(epochs=2, lr=1e-4)
    train_candidate(
        _ScriptedModel(),
        [_batch()],
        [_batch()],
        run_config=config,
        run_dir=tmp_path,
        device="cpu",
    )

    # A completed run is protected from resume before compatibility checks.
    (tmp_path / "result.json").unlink()
    incompatible = _run_config(epochs=2, lr=3e-4)
    with pytest.raises(ResultArtifactError, match="different candidate configuration"):
        train_candidate(
            _ScriptedModel(),
            [_batch()],
            [_batch()],
            run_config=incompatible,
            run_dir=tmp_path,
            device="cpu",
            resume_from=tmp_path / "last.pt",
        )
