from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from unified_bn.training.config import (
    CandidateRunConfig,
    ValidationSelectionRule,
)
from unified_bn.training.metrics import BinaryClassificationMetrics

# Tolerance for the restored-checkpoint consistency guard. After training, the
# trainer reloads ``best.pt`` and re-evaluates it on the validation split; the
# guard asks whether that re-evaluation reproduces ``best_selection_value``,
# i.e. whether the saved checkpoint really is the selected model. It is a
# consistency check, not a bit-exactness test: on GPU the same forward pass is
# not bitwise reproducible across processes (cuDNN RNN kernels for the BiGRU,
# cuBLAS reduction order), and the observed drift is ~1e-7 (static BQN) to
# ~1e-5 (BiGRU) in validation loss. A genuine checkpoint mix-up changes the
# validation loss by orders of magnitude more than this tolerance.
RESTORED_METRIC_ABS_TOL = 1e-4
from unified_bn.training.results import CandidateTrainingResult
from unified_bn.training.search_space import candidate_fingerprint


SELECTION_SCHEMA_NAME = "unified_bn.training.validation_search_selection"
SELECTION_SCHEMA_VERSION = 1


class ValidationSearchError(RuntimeError):
    """Raised when validation-only hyperparameter selection is unsafe."""


@dataclass(frozen=True)
class ValidationSearchSelection:
    """
    Auditable outcome of validation-only hyperparameter selection.

    The held-out test set is intentionally absent from this contract.
    """

    n_candidates: int
    selected_index: int
    selected_fingerprint: str
    selection_metric: str
    selection_mode: str
    selection_value: float
    selected_run_config: Mapping[str, Any]
    selected_best_checkpoint_path: str
    selected_result_path: str

    def __post_init__(self) -> None:
        if isinstance(self.n_candidates, bool) or not isinstance(
            self.n_candidates, int
        ):
            raise TypeError("n_candidates must be an integer.")
        if self.n_candidates <= 0:
            raise ValueError("n_candidates must be positive.")

        if isinstance(self.selected_index, bool) or not isinstance(
            self.selected_index, int
        ):
            raise TypeError("selected_index must be an integer.")
        if not 0 <= self.selected_index < self.n_candidates:
            raise ValueError(
                "selected_index must identify one of the candidate entries."
            )

        if (
            not isinstance(self.selected_fingerprint, str)
            or len(self.selected_fingerprint) != 64
        ):
            raise ValueError(
                "selected_fingerprint must be a 64-character SHA-256 hex digest."
            )
        try:
            int(self.selected_fingerprint, 16)
        except ValueError as exc:
            raise ValueError(
                "selected_fingerprint must be hexadecimal."
            ) from exc

        if (
            not isinstance(self.selection_metric, str)
            or not self.selection_metric.strip()
        ):
            raise ValueError("selection_metric must be a non-empty string.")
        if self.selection_mode not in {"min", "max"}:
            raise ValueError("selection_mode must be 'min' or 'max'.")
        if not math.isfinite(float(self.selection_value)):
            raise ValueError("selection_value must be finite.")

        if not isinstance(self.selected_run_config, Mapping):
            raise TypeError("selected_run_config must be a mapping.")
        _require_json_serializable(self.selected_run_config)

        if (
            not isinstance(self.selected_best_checkpoint_path, str)
            or not self.selected_best_checkpoint_path.strip()
        ):
            raise ValueError(
                "selected_best_checkpoint_path must be a non-empty string."
            )
        if (
            not isinstance(self.selected_result_path, str)
            or not self.selected_result_path.strip()
        ):
            raise ValueError(
                "selected_result_path must be a non-empty string."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": SELECTION_SCHEMA_NAME,
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "validation_selection_complete",
            "n_candidates": int(self.n_candidates),
            "selected_index": int(self.selected_index),
            "selected_fingerprint": self.selected_fingerprint,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "selection_value": float(self.selection_value),
            "selected_run_config": dict(self.selected_run_config),
            "selected_best_checkpoint_path": self.selected_best_checkpoint_path,
            "selected_result_path": self.selected_result_path,
        }
        _require_no_test_keys(payload)
        _require_json_serializable(payload)
        return payload


def select_validation_winner(
    expected_candidates: Sequence[CandidateRunConfig],
    results: Sequence[CandidateTrainingResult],
) -> ValidationSearchSelection:
    """
    Select one hyperparameter candidate using validation information only.

    Safety properties
    -----------------
    * Every frozen/expected candidate must have exactly one completed result.
    * Extra or duplicate results are rejected.
    * Any result carrying held-out test metrics is rejected.
    * All candidates must use one common validation-selection rule.
    * Each result must match its exact expected ``CandidateRunConfig``.
    * The result's restored validation metric must agree with the recorded
      best-selection value.
    * Ties retain the first candidate in the frozen expected-candidate order,
      making selection deterministic and independent of job completion order.

    This function never trains a model and has no test-loader argument.
    """

    candidates = _validated_candidates(expected_candidates)
    completed = _validated_results(results)

    common_rule = _require_common_selection_rule(candidates)

    expected_by_payload: dict[str, tuple[int, CandidateRunConfig]] = {}
    for index, candidate in enumerate(candidates):
        key = _canonical_json(candidate.to_dict())
        if key in expected_by_payload:
            raise ValidationSearchError(
                "Expected candidate list contains duplicate configurations."
            )
        expected_by_payload[key] = (index, candidate)

    result_by_index: dict[int, CandidateTrainingResult] = {}
    for result in completed:
        if result.test_metrics is not None:
            raise ValidationSearchError(
                "Hyperparameter selection refuses results containing held-out "
                "test metrics."
            )

        result_key = _canonical_json(result.run_config)
        expected = expected_by_payload.get(result_key)
        if expected is None:
            raise ValidationSearchError(
                "Completed result does not belong to the frozen expected "
                "candidate set."
            )

        index, candidate = expected
        if index in result_by_index:
            raise ValidationSearchError(
                "Multiple completed results were supplied for the same "
                "candidate configuration."
            )

        _validate_result_against_candidate(
            result,
            candidate=candidate,
            common_rule=common_rule,
        )
        result_by_index[index] = result

    missing = [
        index
        for index in range(len(candidates))
        if index not in result_by_index
    ]
    if missing:
        raise ValidationSearchError(
            "Validation search is incomplete; refusing to select from a "
            "partial candidate set. Missing candidate indices: "
            + ", ".join(str(index) for index in missing)
        )

    incumbent_index = 0
    incumbent_result = result_by_index[0]
    incumbent_value = float(incumbent_result.best_selection_value)

    for index in range(1, len(candidates)):
        result = result_by_index[index]
        value = float(result.best_selection_value)
        if common_rule.is_better(value, incumbent_value):
            incumbent_index = index
            incumbent_result = result
            incumbent_value = value

    selected_candidate = candidates[incumbent_index]

    return ValidationSearchSelection(
        n_candidates=len(candidates),
        selected_index=incumbent_index,
        selected_fingerprint=candidate_fingerprint(selected_candidate),
        selection_metric=common_rule.metric,
        selection_mode=common_rule.mode,
        selection_value=incumbent_value,
        selected_run_config=selected_candidate.to_dict(),
        selected_best_checkpoint_path=incumbent_result.best_checkpoint_path,
        selected_result_path=incumbent_result.result_path,
    )


def write_validation_selection_json(
    path: str | Path,
    selection: ValidationSearchSelection,
) -> Path:
    """Atomically persist a validation-only hyperparameter-selection record."""

    if not isinstance(selection, ValidationSearchSelection):
        raise TypeError(
            "selection must be ValidationSearchSelection, got "
            f"{type(selection)!r}."
        )

    destination = Path(path)
    payload = selection.to_dict()
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

    return destination


def _validated_candidates(
    values: Sequence[CandidateRunConfig],
) -> tuple[CandidateRunConfig, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("expected_candidates must be a finite sequence.")

    candidates = tuple(values)
    if not candidates:
        raise ValidationSearchError(
            "expected_candidates must contain at least one candidate."
        )

    for candidate in candidates:
        if not isinstance(candidate, CandidateRunConfig):
            raise TypeError(
                "Every expected candidate must be CandidateRunConfig; got "
                f"{type(candidate)!r}."
            )

    payloads = [_canonical_json(candidate.to_dict()) for candidate in candidates]
    if len(payloads) != len(set(payloads)):
        raise ValidationSearchError(
            "Expected candidate list contains duplicate configurations."
        )

    return candidates


def _validated_results(
    values: Sequence[CandidateTrainingResult],
) -> tuple[CandidateTrainingResult, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("results must be a finite sequence.")

    results = tuple(values)
    if not results:
        raise ValidationSearchError(
            "results must contain at least one completed candidate."
        )

    for result in results:
        if not isinstance(result, CandidateTrainingResult):
            raise TypeError(
                "Every result must be CandidateTrainingResult; got "
                f"{type(result)!r}."
            )

    return results


def _require_common_selection_rule(
    candidates: Sequence[CandidateRunConfig],
) -> ValidationSelectionRule:
    first = candidates[0].training.checkpoint_selection

    for candidate in candidates[1:]:
        rule = candidate.training.checkpoint_selection
        if rule.metric != first.metric or rule.mode != first.mode:
            raise ValidationSearchError(
                "Hyperparameter candidates use different validation-selection "
                "rules. One search must compare candidates using the same "
                "metric and mode."
            )

    return first


def _validate_result_against_candidate(
    result: CandidateTrainingResult,
    *,
    candidate: CandidateRunConfig,
    common_rule: ValidationSelectionRule,
) -> None:
    if result.selection_metric != common_rule.metric:
        raise ValidationSearchError(
            "Candidate result selection_metric disagrees with the frozen "
            "validation-selection rule."
        )
    if result.selection_mode != common_rule.mode:
        raise ValidationSearchError(
            "Candidate result selection_mode disagrees with the frozen "
            "validation-selection rule."
        )

    value = float(result.best_selection_value)
    if not math.isfinite(value):
        raise ValidationSearchError(
            "Candidate best_selection_value must be finite."
        )

    if _canonical_json(result.run_config) != _canonical_json(
        candidate.to_dict()
    ):
        raise ValidationSearchError(
            "Candidate result run_config does not match its frozen expected "
            "configuration."
        )

    restored_value = _metric_value(
        common_rule,
        result.restored_validation_metrics,
    )
    if not math.isclose(
        value,
        restored_value,
        rel_tol=0.0,
        abs_tol=RESTORED_METRIC_ABS_TOL,
    ):
        raise ValidationSearchError(
            "Candidate best_selection_value disagrees with the restored "
            f"validation metric: |{value!r} - {restored_value!r}| exceeds "
            f"{RESTORED_METRIC_ABS_TOL:g}."
        )


def _metric_value(
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
        raise ValidationSearchError(
            "Unsupported validation-selection metric: "
            f"{rule.metric!r}."
        )

    value = float(values[name])
    if not math.isfinite(value):
        raise ValidationSearchError(
            f"Restored validation metric {rule.metric!r} is not finite."
        )
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationSearchError(
            "Search configuration/result must be finite and JSON serializable."
        ) from exc


def _require_no_test_keys(payload: Any) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if "test" in str(key).lower():
                raise ValidationSearchError(
                    "Validation-selection artifact must not contain test fields."
                )
            _require_no_test_keys(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _require_no_test_keys(value)


def _require_json_serializable(payload: Mapping[str, Any]) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Validation-selection artifact must be finite and JSON serializable."
        ) from exc
