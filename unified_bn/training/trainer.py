from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from unified_bn.data.collate import StaticBatch, TemporalBatch
from unified_bn.training.checkpoint import (
    CheckpointProgress,
    collect_runtime_provenance,
    load_training_checkpoint,
    read_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from unified_bn.training.config import CandidateRunConfig, ValidationSelectionRule
from unified_bn.training.epoch import LossFunction, PipelineBatch, run_binary_epoch
from unified_bn.training.metrics import BinaryClassificationMetrics
from unified_bn.training.optim import build_lr_scheduler, build_optimizer
from unified_bn.training.reproducibility import seed_everything
from unified_bn.training.results import (
    CandidateTrainingResult,
    EpochRecord,
    ResultArtifactError,
    attach_final_test_metrics,
    read_history_json,
    write_candidate_result,
    write_history_json,
)


class TrainerError(RuntimeError):
    """Raised when Stage 6D trainer orchestration violates its protocol."""


def train_candidate(
    model: nn.Module,
    train_batches: Iterable[PipelineBatch],
    val_batches: Iterable[PipelineBatch],
    *,
    run_config: CandidateRunConfig,
    run_dir: str | Path,
    device: torch.device | str,
    loss_fn: Optional[LossFunction] = None,
    dataloader_generator: Optional[torch.Generator] = None,
    resume_from: Optional[str | Path] = None,
    user_provenance: Optional[Mapping[str, Any]] = None,
    non_blocking: bool = False,
) -> CandidateTrainingResult:
    """
    Train one candidate configuration using validation-only model selection.

    Scientific boundary
    -------------------
    This function trains exactly one candidate. It contains no hyperparameter
    grid, no split generation, no HDF5 access, no BQN mathematics, and no test
    loader. Hyperparameter search is an outer orchestration layer that will
    call this function repeatedly with different ``CandidateRunConfig`` values.

    Atlas ABIDE defaults are supplied by Stage 6A configuration (Adam,
    lr=1e-4, weight_decay=1e-4, cosine decay toward 1e-5, 200 epochs), but
    none of those values is hard-coded here. Optimizer, LR, weight decay,
    schedule, schedule horizon, and epoch budget are read from the candidate
    configuration, so they can later become tuning dimensions without
    rewriting the trainer.

    Model selection
    ---------------
    Checkpoint selection is validation-only and uses
    ``run_config.training.checkpoint_selection``. For the Atlas ABIDE protocol
    this is minimum validation loss. Early stopping is a separate policy and
    is disabled by default; with the Atlas defaults the full epoch budget runs.

    Resume
    ------
    ``last.pt`` is the operational resume state; ``best.pt`` is the scientific
    model-selection checkpoint. Stage 6B compatibility checks reject a resume
    from a different seed, split, mode, architecture, optimizer, LR schedule,
    or other candidate configuration.

    Reproducibility note
    --------------------
    This function seeds all training-time RNGs before iteration. For model
    initialization itself to be seed-controlled, callers must invoke the Stage
    6A seed utility before constructing ``model``. Stage 6E/6F integration
    will enforce that construction order for production runs.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}.")
    if not isinstance(run_config, CandidateRunConfig):
        raise TypeError(
            "run_config must be CandidateRunConfig, got "
            f"{type(run_config)!r}."
        )
    if not isinstance(non_blocking, bool):
        raise TypeError("non_blocking must be a bool.")

    training_config = run_config.training
    resolved_device = torch.device(device)
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.json"
    result_path = output_dir / "result.json"

    if resume_from is None:
        collisions = [
            path
            for path in (best_path, last_path, history_path, result_path)
            if path.exists()
        ]
        if collisions:
            raise TrainerError(
                "Fresh candidate run refuses to overwrite existing Stage 6D "
                "artifacts: " + ", ".join(str(path) for path in collisions)
            )
    else:
        if result_path.exists():
            raise TrainerError(
                "Candidate already has result.json; refusing to resume a "
                "validation-complete run."
            )

    _require_reiterable(train_batches, name="train_batches", epochs=training_config.epochs)
    _require_reiterable(val_batches, name="val_batches", epochs=training_config.epochs)

    determinism = training_config.determinism
    seed_everything(
        run_config.identity.seed,
        deterministic_algorithms=determinism.deterministic_algorithms,
        warn_only=determinism.warn_only,
        num_threads=determinism.num_threads,
    )
    if resume_from is None and dataloader_generator is not None:
        dataloader_generator.manual_seed(run_config.identity.seed)

    model.to(resolved_device)
    optimizer = build_optimizer(model, training_config.optimizer)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule_config=training_config.lr_schedule,
        total_epochs=training_config.epochs,
    )

    criterion: LossFunction
    if loss_fn is None:
        criterion = nn.CrossEntropyLoss()
    else:
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable.")
        criterion = loss_fn
    if isinstance(criterion, nn.Module):
        criterion.to(resolved_device)

    provenance = dict(user_provenance or {})
    records: list[EpochRecord] = []
    start_epoch = 1
    best_epoch = 0
    best_selection_value = _initial_incumbent(training_config.checkpoint_selection)

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not history_path.is_file():
            raise TrainerError(
                "Resume requires history.json in the run directory so prior "
                "validation/model-selection decisions remain auditable."
            )

        records = read_history_json(
            history_path,
            expected_run_config=run_config,
        )
        progress = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_run_config=run_config,
            map_location=resolved_device,
            restore_rng=True,
            dataloader_generator=dataloader_generator,
        )

        if len(records) < progress.completed_epoch:
            raise TrainerError(
                "Training history is behind the authoritative resume checkpoint: "
                f"history_epochs={len(records)}, "
                f"checkpoint_epoch={progress.completed_epoch}."
            )
        if len(records) > progress.completed_epoch:
            # Atomic artifacts cannot be committed as one transaction. History
            # is written before last.pt, so a killed job may leave history one
            # epoch ahead. The checkpoint is authoritative for resume; safely
            # discard only the uncommitted suffix and rewrite atomically.
            records = records[: progress.completed_epoch]
            write_history_json(
                history_path,
                run_config=run_config,
                records=records,
            )

        if records[-1].epoch != progress.completed_epoch:
            raise TrainerError(
                "History/checkpoint completed-epoch mismatch during resume."
            )

        best_epoch = progress.best_epoch
        best_selection_value = progress.best_selection_value
        _validate_progress_against_history(
            records,
            selection_rule=training_config.checkpoint_selection,
            best_epoch=best_epoch,
            best_value=best_selection_value,
        )
        start_epoch = progress.completed_epoch + 1

    early_best, early_bad_epochs = _replay_early_stopping_state(
        records,
        run_config=run_config,
    )

    completed_epochs = len(records)
    stopped_early = False

    for epoch in range(start_epoch, training_config.epochs + 1):
        epoch_index = epoch - 1
        learning_rates = scheduler.set_epoch(epoch_index)

        train_metrics = run_binary_epoch(
            model,
            train_batches,
            loss_fn=criterion,
            device=resolved_device,
            training=True,
            optimizer=optimizer,
            non_blocking=non_blocking,
        )
        val_metrics = run_binary_epoch(
            model,
            val_batches,
            loss_fn=criterion,
            device=resolved_device,
            training=False,
            optimizer=None,
            non_blocking=non_blocking,
        )

        record = EpochRecord(
            epoch=epoch,
            learning_rates=tuple(float(x) for x in learning_rates),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        records.append(record)
        completed_epochs = epoch

        selection_value = _validation_metric_value(
            training_config.checkpoint_selection,
            val_metrics,
        )
        improved = training_config.checkpoint_selection.is_better(
            selection_value,
            best_selection_value,
        )
        if improved:
            best_selection_value = float(selection_value)
            best_epoch = int(epoch)

        # Write auditable epoch history before the resume checkpoint. If a job
        # dies between these atomic writes, resume truncates any history suffix
        # beyond the authoritative last.pt epoch and reruns it.
        write_history_json(
            history_path,
            run_config=run_config,
            records=records,
        )

        progress = CheckpointProgress(
            completed_epoch=epoch,
            best_epoch=best_epoch,
            best_selection_value=best_selection_value,
        )

        if improved:
            save_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                run_config=run_config,
                progress=progress,
                dataloader_generator=dataloader_generator,
                user_provenance=provenance,
            )

        save_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_config=run_config,
            progress=progress,
            dataloader_generator=dataloader_generator,
            user_provenance=provenance,
        )

        if training_config.early_stopping.enabled:
            early_value = _validation_metric_value(
                training_config.early_stopping.rule,
                val_metrics,
            )
            early_best, early_bad_epochs, early_improved = _update_early_stopping_state(
                candidate=early_value,
                incumbent=early_best,
                bad_epochs=early_bad_epochs,
                rule=training_config.early_stopping.rule,
                min_delta=training_config.early_stopping.min_delta,
            )
            if (
                not early_improved
                and early_bad_epochs >= int(training_config.early_stopping.patience)
            ):
                stopped_early = True
                break

    if completed_epochs < 1 or best_epoch < 1:
        raise TrainerError("Candidate training completed no selectable epoch.")
    if not best_path.is_file():
        raise TrainerError("Best checkpoint is missing after candidate training.")
    if not last_path.is_file():
        raise TrainerError("Last/resume checkpoint is missing after candidate training.")

    best_checkpoint = read_training_checkpoint(
        best_path,
        expected_run_config=run_config,
        map_location=resolved_device,
    )
    if best_checkpoint.progress.completed_epoch != best_epoch:
        raise TrainerError(
            "best.pt is inconsistent with validation history: "
            f"checkpoint_epoch={best_checkpoint.progress.completed_epoch}, "
            f"selected_best_epoch={best_epoch}."
        )
    if best_checkpoint.progress.best_epoch != best_epoch:
        raise TrainerError("best.pt best_epoch is inconsistent with selected history.")
    if not math.isclose(
        float(best_checkpoint.progress.best_selection_value),
        float(best_selection_value),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise TrainerError(
            "best.pt selection value is inconsistent with selected history."
        )

    # Restore the validation-selected model before any downstream evaluation.
    restore_training_checkpoint(
        best_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=False,
        dataloader_generator=dataloader_generator,
    )

    restored_val_metrics = run_binary_epoch(
        model,
        val_batches,
        loss_fn=criterion,
        device=resolved_device,
        training=False,
        optimizer=None,
        non_blocking=non_blocking,
    )

    best_record = records[best_epoch - 1]
    result = CandidateTrainingResult(
        run_config=run_config.to_dict(),
        requested_epochs=training_config.epochs,
        completed_epochs=completed_epochs,
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        selection_metric=training_config.checkpoint_selection.metric,
        selection_mode=training_config.checkpoint_selection.mode,
        best_selection_value=best_selection_value,
        best_validation_metrics=best_record.val_metrics,
        restored_validation_metrics=restored_val_metrics,
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        history_path=str(history_path),
        result_path=str(result_path),
        loss_function=_qualified_callable_name(criterion),
        runtime_provenance=collect_runtime_provenance(),
        user_provenance=provenance,
        test_metrics=None,
    )
    write_candidate_result(result)
    return result


def evaluate_final_test(
    model: nn.Module,
    test_batches: Iterable[PipelineBatch],
    *,
    device: torch.device | str,
    loss_fn: Optional[LossFunction] = None,
    non_blocking: bool = False,
) -> BinaryClassificationMetrics:
    """
    Explicit held-out test evaluation.

    This function is intentionally separate from ``train_candidate``. The
    trainer has no test-loader argument, so training, checkpoint selection,
    early stopping, and hyperparameter selection cannot inspect held-out test
    data through the Stage 6D API.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}.")

    criterion: LossFunction
    if loss_fn is None:
        criterion = nn.CrossEntropyLoss()
    else:
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable.")
        criterion = loss_fn
    resolved_device = torch.device(device)
    model.to(resolved_device)
    if isinstance(criterion, nn.Module):
        criterion.to(resolved_device)

    return run_binary_epoch(
        model,
        test_batches,
        loss_fn=criterion,
        device=resolved_device,
        training=False,
        optimizer=None,
        non_blocking=non_blocking,
    )


def evaluate_final_test_and_record(
    model: nn.Module,
    test_batches: Iterable[PipelineBatch],
    *,
    result_path: str | Path,
    device: torch.device | str,
    loss_fn: Optional[LossFunction] = None,
    non_blocking: bool = False,
) -> BinaryClassificationMetrics:
    """Explicitly evaluate held-out test data and atomically record it once."""

    metrics = evaluate_final_test(
        model,
        test_batches,
        device=device,
        loss_fn=loss_fn,
        non_blocking=non_blocking,
    )
    attach_final_test_metrics(
        result_path,
        test_metrics=metrics,
    )
    return metrics


def _validation_metric_value(
    rule: ValidationSelectionRule,
    metrics: BinaryClassificationMetrics,
) -> float:
    name = rule.metric.strip().lower()
    values = {
        "val_loss": metrics.loss,
        "val_auc": metrics.roc_auc,
        "val_roc_auc": metrics.roc_auc,
        "val_accuracy": metrics.accuracy,
        "val_acc": metrics.accuracy,
        "val_sensitivity": metrics.sensitivity,
        "val_sens": metrics.sensitivity,
        "val_specificity": metrics.specificity,
        "val_spec": metrics.specificity,
    }
    if name not in values:
        raise TrainerError(
            "Unsupported validation selection metric for binary trainer: "
            f"{rule.metric!r}."
        )
    value = float(values[name])
    if not math.isfinite(value):
        raise TrainerError(
            f"Validation selection metric {rule.metric!r} is not finite."
        )
    return value


def _initial_incumbent(rule: ValidationSelectionRule) -> float:
    if rule.mode == "min":
        return math.inf
    if rule.mode == "max":
        return -math.inf
    raise TrainerError(f"Unsupported validation selection mode: {rule.mode!r}.")


def _validate_progress_against_history(
    records: list[EpochRecord],
    *,
    selection_rule: ValidationSelectionRule,
    best_epoch: int,
    best_value: float,
) -> None:
    if not records:
        raise TrainerError("Resume checkpoint has progress but history is empty.")
    incumbent = _initial_incumbent(selection_rule)
    selected_epoch = 0
    for record in records:
        value = _validation_metric_value(selection_rule, record.val_metrics)
        if selection_rule.is_better(value, incumbent):
            incumbent = value
            selected_epoch = record.epoch

    if selected_epoch != best_epoch or not math.isclose(
        float(incumbent),
        float(best_value),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise TrainerError(
            "Resume checkpoint best-selection progress disagrees with history."
        )


def _replay_early_stopping_state(
    records: list[EpochRecord],
    *,
    run_config: CandidateRunConfig,
) -> tuple[float, int]:
    early = run_config.training.early_stopping
    incumbent = _initial_incumbent(early.rule)
    bad_epochs = 0
    if not early.enabled:
        return incumbent, bad_epochs

    for record in records:
        value = _validation_metric_value(early.rule, record.val_metrics)
        incumbent, bad_epochs, _ = _update_early_stopping_state(
            candidate=value,
            incumbent=incumbent,
            bad_epochs=bad_epochs,
            rule=early.rule,
            min_delta=early.min_delta,
        )
    return incumbent, bad_epochs


def _update_early_stopping_state(
    *,
    candidate: float,
    incumbent: float,
    bad_epochs: int,
    rule: ValidationSelectionRule,
    min_delta: float,
) -> tuple[float, int, bool]:
    if rule.mode == "min":
        improved = candidate < incumbent - min_delta
    elif rule.mode == "max":
        improved = candidate > incumbent + min_delta
    else:
        raise TrainerError(f"Unsupported early-stopping mode: {rule.mode!r}.")

    if improved:
        return float(candidate), 0, True
    return float(incumbent), int(bad_epochs) + 1, False


def _require_reiterable(
    batches: Iterable[PipelineBatch],
    *,
    name: str,
    epochs: int,
) -> None:
    if epochs <= 1:
        return
    try:
        iterator = iter(batches)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable.") from exc
    if iterator is batches:
        raise TrainerError(
            f"{name} is a one-shot iterator but candidate training requires "
            f"{epochs} epochs. Supply a re-iterable DataLoader/collection."
        )


def _qualified_callable_name(obj: Any) -> str:
    cls = type(obj)
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    module = getattr(obj, "__module__", "")
    qualname = getattr(obj, "__qualname__", repr(obj))
    return f"{module}.{qualname}" if module else str(qualname)
