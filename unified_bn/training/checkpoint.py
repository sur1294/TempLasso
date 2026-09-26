from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from unified_bn.data.splits import FrozenSplit, load_frozen_split, validate_frozen_split
from unified_bn.training.config import CandidateRunConfig


CHECKPOINT_SCHEMA_NAME = "unified_bn.training.checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
_SPLIT_FINGERPRINT_SCHEMA = "unified_bn.frozen_split.v1"


class CheckpointError(RuntimeError):
    """Base class for Stage 6 checkpoint failures."""


class CheckpointFormatError(CheckpointError):
    """Raised when a checkpoint is malformed or uses an unsupported schema."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the requested candidate run."""


@dataclass(frozen=True)
class CheckpointProgress:
    """
    Epoch-level state needed to resume a candidate run.

    Epochs are one-based completed epochs, matching the historical BQN logs.
    ``best_epoch`` is the validation-selected epoch observed through
    ``completed_epoch``.
    """

    completed_epoch: int
    best_epoch: int
    best_selection_value: float

    def __post_init__(self) -> None:
        if isinstance(self.completed_epoch, bool) or not isinstance(
            self.completed_epoch, int
        ):
            raise TypeError("completed_epoch must be an integer.")
        if isinstance(self.best_epoch, bool) or not isinstance(self.best_epoch, int):
            raise TypeError("best_epoch must be an integer.")
        if self.completed_epoch < 1:
            raise ValueError("completed_epoch must be >= 1.")
        if self.best_epoch < 1 or self.best_epoch > self.completed_epoch:
            raise ValueError(
                "best_epoch must satisfy 1 <= best_epoch <= completed_epoch."
            )
        if not math.isfinite(float(self.best_selection_value)):
            raise ValueError("best_selection_value must be finite.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_epoch": self.completed_epoch,
            "best_epoch": self.best_epoch,
            "best_selection_value": float(self.best_selection_value),
        }


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    """Validated checkpoint payload that has not yet mutated live objects."""

    path: Path
    progress: CheckpointProgress
    run_config: Mapping[str, Any]
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    scheduler_state: Mapping[str, Any] | None
    rng_state: Mapping[str, Any]
    object_types: Mapping[str, str | None]
    runtime_provenance: Mapping[str, Any]
    user_provenance: Mapping[str, Any]


def frozen_split_identity(split: FrozenSplit) -> str:
    """
    Return a stable semantic fingerprint of an already-validated frozen split.

    Partition names, subject IDs, and *their order* are hashed. Order is part of
    the identity because it can affect deterministic sampling even when split
    membership is unchanged. Labels are intentionally not duplicated here;
    Stage 5 validates dataset labels independently and ``RunIdentity.seed`` is
    stored separately.
    """

    validate_frozen_split(split, require_complete_reference=False)
    payload = {
        "schema": _SPLIT_FINGERPRINT_SCHEMA,
        "train_ids": list(split.train_ids),
        "val_ids": list(split.val_ids),
        "test_ids": list(split.test_ids),
    }
    encoded = _canonical_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def frozen_split_file_identity(
    path: str | Path,
    *,
    reference_subject_ids: list[str] | tuple[str, ...] | None = None,
    require_complete_reference: bool = True,
) -> str:
    """Load a Stage 5 frozen split and return its semantic fingerprint."""

    split = load_frozen_split(
        path,
        reference_subject_ids=reference_subject_ids,
        require_complete_reference=require_complete_reference,
    )
    return frozen_split_identity(split)


def capture_rng_state(
    *,
    dataloader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """
    Capture RNG streams required for epoch-boundary exact resume.

    A dedicated DataLoader generator is a separate RNG stream and therefore
    must be checkpointed explicitly when one is used for shuffling.
    """

    np_state = np.random.get_state()
    numpy_state = {
        "bit_generator": str(np_state[0]),
        "keys": [int(x) for x in np_state[1].tolist()],
        "position": int(np_state[2]),
        "has_gauss": int(np_state[3]),
        "cached_gaussian": float(np_state[4]),
    }

    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]

    loader_state = None
    if dataloader_generator is not None:
        loader_state = dataloader_generator.get_state().cpu()

    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": cuda_states,
        "dataloader_generator": loader_state,
    }


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    dataloader_generator: torch.Generator | None = None,
) -> None:
    """
    Restore RNG streams captured by :func:`capture_rng_state`.

    If CUDA RNG states were saved but the current runtime cannot reproduce the
    same CUDA device count, exact resume is impossible and the function fails
    loudly instead of silently degrading reproducibility.
    """

    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "dataloader_generator",
    }
    missing = required - set(state)
    if missing:
        raise CheckpointFormatError(
            f"RNG state is missing required fields: {sorted(missing)}."
        )

    random.setstate(state["python"])

    np_payload = state["numpy"]
    if not isinstance(np_payload, Mapping):
        raise CheckpointFormatError("NumPy RNG state must be a mapping.")
    np_required = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    np_missing = np_required - set(np_payload)
    if np_missing:
        raise CheckpointFormatError(
            f"NumPy RNG state is missing fields: {sorted(np_missing)}."
        )

    np.random.set_state(
        (
            str(np_payload["bit_generator"]),
            np.asarray(np_payload["keys"], dtype=np.uint32),
            int(np_payload["position"]),
            int(np_payload["has_gauss"]),
            float(np_payload["cached_gaussian"]),
        )
    )

    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor):
        raise CheckpointFormatError("Torch CPU RNG state must be a tensor.")
    torch.set_rng_state(cpu_state.cpu())

    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise CheckpointFormatError("Torch CUDA RNG state must be a list.")
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable; "
                "exact resume is not possible."
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointCompatibilityError(
                "CUDA device count differs from checkpoint RNG state: "
                f"checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}."
            )
        torch.cuda.set_rng_state_all([state_tensor.cpu() for state_tensor in cuda_states])

    loader_state = state["dataloader_generator"]
    if loader_state is not None:
        if dataloader_generator is None:
            raise CheckpointCompatibilityError(
                "Checkpoint contains a DataLoader generator RNG state but no "
                "generator was supplied for restoration."
            )
        if not isinstance(loader_state, torch.Tensor):
            raise CheckpointFormatError(
                "DataLoader generator RNG state must be a tensor or None."
            )
        dataloader_generator.set_state(loader_state.cpu())


def collect_runtime_provenance() -> dict[str, Any]:
    """Collect software/runtime metadata without invoking external commands."""

    cudnn_version = None
    if torch.backends.cudnn.is_available():
        cudnn_version = torch.backends.cudnn.version()

    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    run_config: CandidateRunConfig,
    progress: CheckpointProgress,
    dataloader_generator: torch.Generator | None = None,
    user_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """
    Atomically save a complete candidate-training resume checkpoint.

    The checkpoint is architecture-agnostic: it stores pipeline state, not BQN
    internals. Test metrics are deliberately absent from the schema.
    """

    run_payload = run_config.to_dict()
    provenance_payload = dict(user_provenance or {})
    _require_json_serializable(provenance_payload, name="user_provenance")

    payload = {
        "schema_name": CHECKPOINT_SCHEMA_NAME,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_config": run_payload,
        "progress": progress.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "rng_state": capture_rng_state(
            dataloader_generator=dataloader_generator
        ),
        "object_types": {
            "model": _qualified_type_name(model),
            "optimizer": _qualified_type_name(optimizer),
            "scheduler": None if scheduler is None else _qualified_type_name(scheduler),
        },
        "runtime_provenance": collect_runtime_provenance(),
        "user_provenance": provenance_payload,
    }

    destination = Path(path)
    _atomic_torch_save(payload, destination)
    return destination


def read_training_checkpoint(
    path: str | Path,
    *,
    expected_run_config: CandidateRunConfig,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
) -> LoadedTrainingCheckpoint:
    """
    Read and validate checkpoint format + scientific run compatibility.

    No model, optimizer, scheduler, or RNG object is mutated by this function.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:  # torch raises several serialization-specific types
        raise CheckpointFormatError(
            f"Could not safely read checkpoint: {checkpoint_path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise CheckpointFormatError("Checkpoint root must be a mapping.")

    _validate_checkpoint_schema(payload)
    _validate_run_compatibility(
        checkpoint_run_config=payload["run_config"],
        expected_run_config=expected_run_config.to_dict(),
    )

    progress_payload = payload["progress"]
    if not isinstance(progress_payload, Mapping):
        raise CheckpointFormatError("Checkpoint progress must be a mapping.")
    try:
        progress = CheckpointProgress(
            completed_epoch=int(progress_payload["completed_epoch"]),
            best_epoch=int(progress_payload["best_epoch"]),
            best_selection_value=float(progress_payload["best_selection_value"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError("Checkpoint progress is malformed.") from exc

    object_types = payload["object_types"]
    if not isinstance(object_types, Mapping):
        raise CheckpointFormatError("Checkpoint object_types must be a mapping.")

    return LoadedTrainingCheckpoint(
        path=checkpoint_path,
        progress=progress,
        run_config=payload["run_config"],
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        rng_state=payload["rng_state"],
        object_types=object_types,
        runtime_provenance=payload["runtime_provenance"],
        user_provenance=payload["user_provenance"],
    )


def restore_training_checkpoint(
    checkpoint: LoadedTrainingCheckpoint,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    restore_rng: bool = True,
    dataloader_generator: torch.Generator | None = None,
    strict_model: bool = True,
) -> CheckpointProgress:
    """
    Restore validated state into live training objects.

    Object-type and scheduler-presence checks happen before any live state is
    mutated. This prevents a known-incompatible resume from partially altering
    the candidate run.
    """

    expected_types = checkpoint.object_types
    actual_types = {
        "model": _qualified_type_name(model),
        "optimizer": _qualified_type_name(optimizer),
        "scheduler": None if scheduler is None else _qualified_type_name(scheduler),
    }

    type_differences = []
    for key in ("model", "optimizer", "scheduler"):
        if expected_types.get(key) != actual_types[key]:
            type_differences.append(
                f"{key}: checkpoint={expected_types.get(key)!r}, "
                f"current={actual_types[key]!r}"
            )
    if type_differences:
        raise CheckpointCompatibilityError(
            "Training object types are incompatible with checkpoint:\n  - "
            + "\n  - ".join(type_differences)
        )

    checkpoint_has_scheduler = checkpoint.scheduler_state is not None
    current_has_scheduler = scheduler is not None
    if checkpoint_has_scheduler != current_has_scheduler:
        raise CheckpointCompatibilityError(
            "Scheduler presence differs between checkpoint and current run."
        )

    model.load_state_dict(checkpoint.model_state, strict=strict_model)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint.scheduler_state)

    if restore_rng:
        restore_rng_state(
            checkpoint.rng_state,
            dataloader_generator=dataloader_generator,
        )

    return checkpoint.progress


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    expected_run_config: CandidateRunConfig,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
    restore_rng: bool = True,
    dataloader_generator: torch.Generator | None = None,
    strict_model: bool = True,
) -> CheckpointProgress:
    """Convenience wrapper: safely read, validate, and then restore."""

    checkpoint = read_training_checkpoint(
        path,
        expected_run_config=expected_run_config,
        map_location=map_location,
    )
    return restore_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=restore_rng,
        dataloader_generator=dataloader_generator,
        strict_model=strict_model,
    )


def _validate_checkpoint_schema(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_name",
        "schema_version",
        "run_config",
        "progress",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "object_types",
        "runtime_provenance",
        "user_provenance",
    }
    missing = required - set(payload)
    if missing:
        raise CheckpointFormatError(
            f"Checkpoint is missing required fields: {sorted(missing)}."
        )

    if payload["schema_name"] != CHECKPOINT_SCHEMA_NAME:
        raise CheckpointFormatError(
            "Unsupported checkpoint schema: "
            f"{payload['schema_name']!r}."
        )
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointFormatError(
            "Unsupported checkpoint schema version: "
            f"{payload['schema_version']!r}; expected {CHECKPOINT_SCHEMA_VERSION}."
        )

    for field_name in (
        "run_config",
        "model_state",
        "optimizer_state",
        "rng_state",
        "object_types",
        "runtime_provenance",
        "user_provenance",
    ):
        if not isinstance(payload[field_name], Mapping):
            raise CheckpointFormatError(
                f"Checkpoint field {field_name!r} must be a mapping."
            )

    if payload["scheduler_state"] is not None and not isinstance(
        payload["scheduler_state"], Mapping
    ):
        raise CheckpointFormatError(
            "Checkpoint scheduler_state must be a mapping or None."
        )


def _validate_run_compatibility(
    *,
    checkpoint_run_config: Any,
    expected_run_config: Mapping[str, Any],
) -> None:
    if not isinstance(checkpoint_run_config, Mapping):
        raise CheckpointFormatError("Checkpoint run_config must be a mapping.")

    differences = _mapping_differences(
        expected_run_config,
        checkpoint_run_config,
        path="run_config",
    )
    if differences:
        preview = differences[:20]
        suffix = "" if len(differences) <= 20 else f"\n  ... {len(differences) - 20} more"
        raise CheckpointCompatibilityError(
            "Checkpoint candidate configuration is incompatible:\n  - "
            + "\n  - ".join(preview)
            + suffix
        )


def _mapping_differences(expected: Any, actual: Any, *, path: str) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        all_keys = sorted(set(expected) | set(actual), key=str)
        for key in all_keys:
            child = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{child}: unexpected checkpoint field")
            elif key not in actual:
                differences.append(f"{child}: missing from checkpoint")
            else:
                differences.extend(
                    _mapping_differences(expected[key], actual[key], path=child)
                )
        return differences

    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return [
                f"{path}: expected length {len(expected)}, checkpoint length {len(actual)}"
            ]
        differences = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            differences.extend(
                _mapping_differences(
                    expected_item,
                    actual_item,
                    path=f"{path}[{index}]",
                )
            )
        return differences

    if expected != actual:
        return [f"{path}: expected {expected!r}, checkpoint {actual!r}"]
    return []


def _qualified_type_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _require_json_serializable(payload: Mapping[str, Any], *, name: str) -> None:
    try:
        _canonical_json(payload)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite and JSON serializable.") from exc


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise
