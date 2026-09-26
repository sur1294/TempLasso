from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from unified_bn.training.config import CandidateRunConfig
from unified_bn.training.metrics import BinaryClassificationMetrics


HISTORY_SCHEMA_NAME = "unified_bn.training.history"
HISTORY_SCHEMA_VERSION = 1
RESULT_SCHEMA_NAME = "unified_bn.training.result"
RESULT_SCHEMA_VERSION = 1


class ResultArtifactError(RuntimeError):
    """Raised when a Stage 6 training/result artifact is malformed or unsafe."""


@dataclass(frozen=True)
class EpochRecord:
    """One completed epoch of train/validation history."""

    epoch: int
    learning_rates: tuple[float, ...]
    train_metrics: BinaryClassificationMetrics
    val_metrics: BinaryClassificationMetrics

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int):
            raise TypeError("epoch must be an integer.")
        if self.epoch < 1:
            raise ValueError("epoch must be >= 1.")
        if not self.learning_rates:
            raise ValueError("learning_rates must be non-empty.")
        for value in self.learning_rates:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    "learning_rates must contain finite nonnegative values."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "learning_rates": [float(x) for x in self.learning_rates],
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpochRecord":
        if not isinstance(payload, Mapping):
            raise ResultArtifactError("Epoch record must be a mapping.")
        try:
            return cls(
                epoch=int(payload["epoch"]),
                learning_rates=tuple(float(x) for x in payload["learning_rates"]),
                train_metrics=_metrics_from_mapping(payload["train_metrics"]),
                val_metrics=_metrics_from_mapping(payload["val_metrics"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultArtifactError("Malformed epoch record.") from exc


@dataclass(frozen=True)
class CandidateTrainingResult:
    """
    Machine-readable result for one candidate configuration.

    During tuning, ``test_metrics`` is deliberately ``None``. The test result
    can only be attached by an explicit final-test action after validation-only
    hyperparameter selection has chosen this candidate.
    """

    run_config: Mapping[str, Any]
    requested_epochs: int
    completed_epochs: int
    stopped_early: bool
    best_epoch: int
    selection_metric: str
    selection_mode: str
    best_selection_value: float
    best_validation_metrics: BinaryClassificationMetrics
    restored_validation_metrics: BinaryClassificationMetrics
    best_checkpoint_path: str
    last_checkpoint_path: str
    history_path: str
    result_path: str
    loss_function: str
    runtime_provenance: Mapping[str, Any]
    user_provenance: Mapping[str, Any]
    test_metrics: Optional[BinaryClassificationMetrics] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": RESULT_SCHEMA_NAME,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": (
                "final_test_complete" if self.test_metrics is not None else "validation_complete"
            ),
            "run_config": dict(self.run_config),
            "requested_epochs": int(self.requested_epochs),
            "completed_epochs": int(self.completed_epochs),
            "stopped_early": bool(self.stopped_early),
            "best_epoch": int(self.best_epoch),
            "selection_metric": str(self.selection_metric),
            "selection_mode": str(self.selection_mode),
            "best_selection_value": float(self.best_selection_value),
            "best_validation_metrics": self.best_validation_metrics.to_dict(),
            "restored_validation_metrics": self.restored_validation_metrics.to_dict(),
            "checkpoints": {
                "best": self.best_checkpoint_path,
                "last": self.last_checkpoint_path,
            },
            "history_path": self.history_path,
            "result_path": self.result_path,
            "loss_function": self.loss_function,
            "runtime_provenance": dict(self.runtime_provenance),
            "user_provenance": dict(self.user_provenance),
            "test_metrics": (
                None if self.test_metrics is None else self.test_metrics.to_dict()
            ),
        }
        _require_json_serializable(payload)
        return payload


def write_history_json(
    path: str | Path,
    *,
    run_config: CandidateRunConfig,
    records: Iterable[EpochRecord],
) -> Path:
    record_list = list(records)
    _validate_contiguous_records(record_list)

    payload = {
        "schema_name": HISTORY_SCHEMA_NAME,
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_config": run_config.to_dict(),
        "records": [record.to_dict() for record in record_list],
    }
    _atomic_json_dump(payload, Path(path))
    return Path(path)


def read_history_json(
    path: str | Path,
    *,
    expected_run_config: CandidateRunConfig,
) -> list[EpochRecord]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Training history not found: {source}")

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultArtifactError(f"Could not read training history: {source}") from exc

    if not isinstance(payload, Mapping):
        raise ResultArtifactError("History root must be a mapping.")
    if payload.get("schema_name") != HISTORY_SCHEMA_NAME:
        raise ResultArtifactError("Unsupported training-history schema.")
    if payload.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise ResultArtifactError("Unsupported training-history schema version.")
    if payload.get("run_config") != expected_run_config.to_dict():
        raise ResultArtifactError(
            "Training history belongs to a different candidate configuration."
        )

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ResultArtifactError("History records must be a list.")
    records = [EpochRecord.from_dict(item) for item in raw_records]
    _validate_contiguous_records(records)
    return records


def write_candidate_result(
    result: CandidateTrainingResult,
) -> Path:
    destination = Path(result.result_path)
    _atomic_json_dump(result.to_dict(), destination)
    return destination


def attach_final_test_metrics(
    result_path: str | Path,
    *,
    test_metrics: BinaryClassificationMetrics,
) -> dict[str, Any]:
    """
    Attach held-out test metrics exactly once to a validation-complete result.

    This function does not evaluate data. It only records metrics produced by
    the explicit final-test evaluation path.
    """

    source = Path(result_path)
    if not source.is_file():
        raise FileNotFoundError(f"Candidate result not found: {source}")

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultArtifactError(f"Could not read candidate result: {source}") from exc

    _validate_result_payload(payload)
    if payload.get("test_metrics") is not None:
        raise ResultArtifactError(
            "Final test metrics are already recorded; refusing to overwrite them."
        )

    payload["test_metrics"] = test_metrics.to_dict()
    payload["status"] = "final_test_complete"
    _atomic_json_dump(payload, source)
    return payload


def _validate_result_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ResultArtifactError("Result root must be a mapping.")
    if payload.get("schema_name") != RESULT_SCHEMA_NAME:
        raise ResultArtifactError("Unsupported result schema.")
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultArtifactError("Unsupported result schema version.")


def _metrics_from_mapping(payload: Any) -> BinaryClassificationMetrics:
    if not isinstance(payload, Mapping):
        raise ResultArtifactError("Metric payload must be a mapping.")
    try:
        return BinaryClassificationMetrics(
            loss=float(payload["loss"]),
            roc_auc=float(payload["roc_auc"]),
            accuracy=float(payload["accuracy"]),
            sensitivity=float(payload["sensitivity"]),
            specificity=float(payload["specificity"]),
            n_samples=int(payload["n_samples"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultArtifactError("Malformed binary metric payload.") from exc


def _validate_contiguous_records(records: list[EpochRecord]) -> None:
    expected = list(range(1, len(records) + 1))
    actual = [record.epoch for record in records]
    if actual != expected:
        raise ResultArtifactError(
            "Training history must contain contiguous one-based epochs; "
            f"got {actual}."
        )


def _require_json_serializable(payload: Mapping[str, Any]) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResultArtifactError(
            "Result artifact must be finite and JSON serializable."
        ) from exc


def _atomic_json_dump(payload: Mapping[str, Any], destination: Path) -> None:
    _require_json_serializable(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)

    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
