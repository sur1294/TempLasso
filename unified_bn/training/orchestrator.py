from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from unified_bn.training.config import CandidateRunConfig
from unified_bn.training.epoch import LossFunction, PipelineBatch
from unified_bn.training.metrics import BinaryClassificationMetrics
from unified_bn.training.reproducibility import seed_everything
from unified_bn.training.results import (
    CandidateTrainingResult,
    RESULT_SCHEMA_NAME,
    RESULT_SCHEMA_VERSION,
)
from unified_bn.training.search import (
    ValidationSearchError,
    ValidationSearchSelection,
    select_validation_winner,
    write_validation_selection_json,
)
from unified_bn.training.search_space import candidate_fingerprint
from unified_bn.training.trainer import train_candidate


class SearchOrchestrationError(RuntimeError):
    """Raised when candidate execution/resume state is unsafe or inconsistent."""


@dataclass(frozen=True)
class CandidateRuntime:
    """
    Runtime objects required to train exactly one already-frozen candidate.

    Model/data construction is deliberately injected from outside this module.
    That keeps search orchestration independent of BQN, BiGRU, ABIDE, HDF5,
    and future backbone implementations.
    """

    model: nn.Module
    train_batches: Iterable[PipelineBatch]
    val_batches: Iterable[PipelineBatch]
    device: torch.device | str
    loss_fn: Optional[LossFunction] = None
    dataloader_generator: Optional[torch.Generator] = None
    user_provenance: Optional[Mapping[str, Any]] = None
    non_blocking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"model must be torch.nn.Module, got {type(self.model)!r}."
            )
        if not isinstance(self.non_blocking, bool):
            raise TypeError("non_blocking must be a bool.")
        if (
            self.dataloader_generator is not None
            and not isinstance(self.dataloader_generator, torch.Generator)
        ):
            raise TypeError(
                "dataloader_generator must be torch.Generator or None."
            )
        if self.user_provenance is not None and not isinstance(
            self.user_provenance, Mapping
        ):
            raise TypeError("user_provenance must be a mapping or None.")


CandidateRuntimeFactory = Callable[
    [CandidateRunConfig, int, Path],
    CandidateRuntime,
]


def candidate_run_directory(
    search_dir: str | Path,
    *,
    index: int,
    candidate: CandidateRunConfig,
) -> Path:
    """
    Deterministic per-candidate directory.

    Frozen candidate order is kept in the numeric prefix; the fingerprint
    protects against accidentally reusing that index for another config.
    """

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer.")
    if index < 0:
        raise ValueError("index must be nonnegative.")
    if not isinstance(candidate, CandidateRunConfig):
        raise TypeError(
            "candidate must be CandidateRunConfig, got "
            f"{type(candidate)!r}."
        )

    fingerprint = candidate_fingerprint(candidate)
    return Path(search_dir) / (
        f"candidate_{index:04d}_{fingerprint[:12]}"
    )


def run_validation_search(
    candidates: Sequence[CandidateRunConfig],
    *,
    search_dir: str | Path,
    runtime_factory: CandidateRuntimeFactory,
) -> ValidationSearchSelection:
    """
    Execute/resume a complete validation-only hyperparameter search.

    Scientific boundaries
    ---------------------
    * Candidate definitions are frozen before execution.
    * Each candidate is trained only through the existing ``train_candidate``.
    * Model initialization is seeded before ``runtime_factory`` is called.
    * Completed validation-only candidates are reused without retraining.
    * Interrupted candidates resume from their deterministic ``last.pt``.
    * A partial/corrupt artifact state is rejected rather than deleted.
    * Held-out test metrics are never evaluated or accepted here.
    * Selection occurs only after every expected candidate is complete.
    * Final selection uses the Stage 6I-A validation-only selector.

    This function has no test-loader argument and never calls the final-test
    evaluation API.
    """

    frozen_candidates = _validate_candidate_sequence(candidates)
    if not callable(runtime_factory):
        raise TypeError("runtime_factory must be callable.")

    _preflight_common_search_identity(frozen_candidates)

    root = Path(search_dir)
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.json"

    if selection_path.exists():
        # A previous completed search is immutable. Reruns must verify/reuse
        # candidate results, then reproduce the same selection rather than
        # silently overwrite the scientific decision.
        existing_selection = _read_selection_payload(selection_path)
    else:
        existing_selection = None

    completed_results: list[CandidateTrainingResult] = []

    for index, candidate in enumerate(frozen_candidates):
        run_dir = candidate_run_directory(
            root,
            index=index,
            candidate=candidate,
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        result_path = run_dir / "result.json"
        last_path = run_dir / "last.pt"
        best_path = run_dir / "best.pt"
        history_path = run_dir / "history.json"

        if result_path.is_file():
            result = _read_validation_candidate_result(
                result_path,
                expected_candidate=candidate,
            )
            _require_completed_artifacts(
                result,
                result_path=result_path,
                best_path=best_path,
                last_path=last_path,
                history_path=history_path,
            )
            completed_results.append(result)
            continue

        _validate_incomplete_artifact_state(
            run_dir=run_dir,
            last_path=last_path,
            best_path=best_path,
            history_path=history_path,
        )

        # train_candidate() seeds training RNGs, but model parameters already
        # exist by then. Seed here first so candidate model initialization is
        # itself reproducible.
        determinism = candidate.training.determinism
        seed_everything(
            candidate.identity.seed,
            deterministic_algorithms=determinism.deterministic_algorithms,
            warn_only=determinism.warn_only,
            num_threads=determinism.num_threads,
        )

        runtime = runtime_factory(candidate, index, run_dir)
        if not isinstance(runtime, CandidateRuntime):
            raise TypeError(
                "runtime_factory must return CandidateRuntime, got "
                f"{type(runtime)!r}."
            )

        resume_from: Optional[Path]
        if last_path.is_file():
            resume_from = last_path
        else:
            resume_from = None

        result = train_candidate(
            runtime.model,
            runtime.train_batches,
            runtime.val_batches,
            run_config=candidate,
            run_dir=run_dir,
            device=runtime.device,
            loss_fn=runtime.loss_fn,
            dataloader_generator=runtime.dataloader_generator,
            resume_from=resume_from,
            user_provenance=runtime.user_provenance,
            non_blocking=runtime.non_blocking,
        )

        _validate_returned_result(
            result,
            expected_candidate=candidate,
            expected_result_path=result_path,
        )
        completed_results.append(result)

    selection = select_validation_winner(
        frozen_candidates,
        completed_results,
    )

    if existing_selection is not None:
        if existing_selection != selection.to_dict():
            raise SearchOrchestrationError(
                "Existing selection.json disagrees with the selection "
                "recomputed from completed validation results. Refusing to "
                "overwrite a prior scientific selection."
            )
    else:
        write_validation_selection_json(selection_path, selection)

    return selection


def _validate_candidate_sequence(
    values: Sequence[CandidateRunConfig],
) -> tuple[CandidateRunConfig, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("candidates must be a finite sequence.")

    candidates = tuple(values)
    if not candidates:
        raise SearchOrchestrationError(
            "candidates must contain at least one frozen candidate."
        )

    fingerprints: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, CandidateRunConfig):
            raise TypeError(
                "Every candidate must be CandidateRunConfig; got "
                f"{type(candidate)!r}."
            )
        fingerprints.append(candidate_fingerprint(candidate))

    if len(fingerprints) != len(set(fingerprints)):
        raise SearchOrchestrationError(
            "Frozen candidate list contains duplicate complete configurations."
        )

    return candidates


def _preflight_common_search_identity(
    candidates: Sequence[CandidateRunConfig],
) -> None:
    first = candidates[0]
    first_rule = first.training.checkpoint_selection

    for candidate in candidates[1:]:
        if candidate.identity.mode != first.identity.mode:
            raise SearchOrchestrationError(
                "One validation search cannot mix training modes."
            )
        if candidate.identity.seed != first.identity.seed:
            raise SearchOrchestrationError(
                "One validation search cannot mix random seeds."
            )
        if candidate.identity.split_identity != first.identity.split_identity:
            raise SearchOrchestrationError(
                "One validation search cannot mix frozen split identities."
            )

        rule = candidate.training.checkpoint_selection
        if (
            rule.metric != first_rule.metric
            or rule.mode != first_rule.mode
        ):
            raise SearchOrchestrationError(
                "One validation search cannot mix validation-selection rules."
            )


def _validate_incomplete_artifact_state(
    *,
    run_dir: Path,
    last_path: Path,
    best_path: Path,
    history_path: Path,
) -> None:
    """
    Accept only two incomplete states:

      fresh:  no Stage 6D artifacts
      resume: last.pt + history.json, with best.pt optional

    Anything else is ambiguous and therefore fail-closed.
    """

    present = {
        "last.pt": last_path.is_file(),
        "best.pt": best_path.is_file(),
        "history.json": history_path.is_file(),
    }

    if not any(present.values()):
        return

    if present["last.pt"] and present["history.json"]:
        return

    raise SearchOrchestrationError(
        "Candidate directory contains an unsafe partial artifact state and "
        "cannot be automatically resumed: "
        f"{run_dir}; present={present}."
    )


def _validate_returned_result(
    result: CandidateTrainingResult,
    *,
    expected_candidate: CandidateRunConfig,
    expected_result_path: Path,
) -> None:
    if not isinstance(result, CandidateTrainingResult):
        raise TypeError(
            "train_candidate must return CandidateTrainingResult, got "
            f"{type(result)!r}."
        )
    if result.test_metrics is not None:
        raise SearchOrchestrationError(
            "Validation search received a result containing held-out test "
            "metrics."
        )
    if result.run_config != expected_candidate.to_dict():
        raise SearchOrchestrationError(
            "train_candidate returned a result for a different candidate."
        )
    if Path(result.result_path).resolve() != expected_result_path.resolve():
        raise SearchOrchestrationError(
            "train_candidate returned an unexpected result.json path."
        )


def _read_validation_candidate_result(
    source: Path,
    *,
    expected_candidate: CandidateRunConfig,
) -> CandidateTrainingResult:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchOrchestrationError(
            f"Could not read completed candidate result: {source}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise SearchOrchestrationError(
            "Completed candidate result root must be a mapping."
        )
    if payload.get("schema_name") != RESULT_SCHEMA_NAME:
        raise SearchOrchestrationError(
            "Completed candidate uses an unsupported result schema."
        )
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise SearchOrchestrationError(
            "Completed candidate uses an unsupported result schema version."
        )
    if payload.get("status") != "validation_complete":
        raise SearchOrchestrationError(
            "Validation search may only reuse validation-complete results."
        )
    if payload.get("test_metrics") is not None:
        raise SearchOrchestrationError(
            "Validation search refuses a completed result containing held-out "
            "test metrics."
        )

    run_config = payload.get("run_config")
    if run_config != expected_candidate.to_dict():
        raise SearchOrchestrationError(
            "Completed result belongs to a different candidate configuration."
        )

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise SearchOrchestrationError(
            "Completed result has malformed checkpoint metadata."
        )

    try:
        result = CandidateTrainingResult(
            run_config=run_config,
            requested_epochs=int(payload["requested_epochs"]),
            completed_epochs=int(payload["completed_epochs"]),
            stopped_early=bool(payload["stopped_early"]),
            best_epoch=int(payload["best_epoch"]),
            selection_metric=str(payload["selection_metric"]),
            selection_mode=str(payload["selection_mode"]),
            best_selection_value=float(payload["best_selection_value"]),
            best_validation_metrics=_metrics_from_payload(
                payload["best_validation_metrics"]
            ),
            restored_validation_metrics=_metrics_from_payload(
                payload["restored_validation_metrics"]
            ),
            best_checkpoint_path=str(checkpoints["best"]),
            last_checkpoint_path=str(checkpoints["last"]),
            history_path=str(payload["history_path"]),
            result_path=str(payload["result_path"]),
            loss_function=str(payload["loss_function"]),
            runtime_provenance=_mapping_copy(
                payload["runtime_provenance"],
                name="runtime_provenance",
            ),
            user_provenance=_mapping_copy(
                payload["user_provenance"],
                name="user_provenance",
            ),
            test_metrics=None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SearchOrchestrationError(
            f"Malformed completed candidate result: {source}"
        ) from exc

    if Path(result.result_path).resolve() != source.resolve():
        raise SearchOrchestrationError(
            "Completed result's embedded result_path does not match its file."
        )

    return result


def _require_completed_artifacts(
    result: CandidateTrainingResult,
    *,
    result_path: Path,
    best_path: Path,
    last_path: Path,
    history_path: Path,
) -> None:
    expected = {
        "best checkpoint": (
            Path(result.best_checkpoint_path),
            best_path,
        ),
        "last checkpoint": (
            Path(result.last_checkpoint_path),
            last_path,
        ),
        "history": (
            Path(result.history_path),
            history_path,
        ),
        "result": (
            Path(result.result_path),
            result_path,
        ),
    }

    for name, (embedded, actual) in expected.items():
        if embedded.resolve() != actual.resolve():
            raise SearchOrchestrationError(
                f"Completed candidate {name} path disagrees with its "
                "deterministic run directory."
            )
        if not actual.is_file():
            raise SearchOrchestrationError(
                f"Completed candidate is missing required {name}: {actual}"
            )


def _metrics_from_payload(payload: Any) -> BinaryClassificationMetrics:
    if not isinstance(payload, Mapping):
        raise SearchOrchestrationError(
            "Binary metric payload must be a mapping."
        )
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
        raise SearchOrchestrationError(
            "Malformed binary metric payload."
        ) from exc


def _mapping_copy(payload: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SearchOrchestrationError(f"{name} must be a mapping.")
    return dict(payload)


def _read_selection_payload(source: Path) -> dict[str, Any]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchOrchestrationError(
            f"Could not read existing validation selection: {source}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise SearchOrchestrationError(
            "Existing selection.json root must be a mapping."
        )
    return dict(payload)
