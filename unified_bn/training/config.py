from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from unified_bn.temporal.windowing import WindowPolicyConfig


TrainingMode = Literal["vanilla", "temporal", "fusion"]
SelectionMode = Literal["min", "max"]
LRScheduleName = Literal["cosine", "constant"]


@dataclass(frozen=True)
class DeterminismConfig:
    """PyTorch reproducibility policy for a candidate training run."""

    deterministic_algorithms: bool = True
    warn_only: bool = False
    num_threads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.deterministic_algorithms, bool):
            raise TypeError("deterministic_algorithms must be a bool.")
        if not isinstance(self.warn_only, bool):
            raise TypeError("warn_only must be a bool.")
        if isinstance(self.num_threads, bool) or not isinstance(
            self.num_threads, int
        ):
            raise TypeError("num_threads must be an integer.")
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for the Atlas binary-classification protocol."""

    name: Literal["adam"] = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("Optimizer learning rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("Optimizer weight_decay must be nonnegative.")


@dataclass(frozen=True)
class LRScheduleConfig:
    """
    Explicit learning-rate schedule contract.

    The Atlas manuscript specifies decay toward 1e-5 without uniquely
    identifying a functional form.  The validated historical BQN trainer
    makes that choice explicit as cosine decay; ``constant`` is retained only
    as a controlled ablation option.
    """

    name: LRScheduleName = "cosine"
    min_lr: float = 1e-5

    def __post_init__(self) -> None:
        if self.min_lr < 0.0:
            raise ValueError("Scheduler min_lr must be nonnegative.")


@dataclass(frozen=True)
class ValidationSelectionRule:
    """
    Validation-only comparison rule.

    The default ``val_loss/min`` rule is the Atlas ABIDE protocol and can be
    reused at two distinct levels:

    1. checkpoint selection within one candidate configuration;
    2. hyperparameter selection across candidate configurations.

    Test metrics are intentionally absent from this contract.
    """

    metric: str = "val_loss"
    mode: SelectionMode = "min"

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("Selection metric must be a non-empty string.")

    def is_better(self, candidate: float, incumbent: float) -> bool:
        """Return True only for a strict improvement; ties do not replace."""
        if self.mode == "min":
            return candidate < incumbent
        if self.mode == "max":
            return candidate > incumbent
        raise RuntimeError(f"Unsupported selection mode: {self.mode!r}")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """
    Optimization-stopping policy, kept separate from checkpoint selection.

    The Atlas ABIDE reproduction path disables early stopping and trains for
    the full epoch budget while retaining the minimum-validation-loss
    checkpoint.
    """

    enabled: bool = False
    patience: int | None = None
    min_delta: float = 0.0
    rule: ValidationSelectionRule = field(
        default_factory=ValidationSelectionRule
    )

    def __post_init__(self) -> None:
        if self.min_delta < 0.0:
            raise ValueError("Early-stopping min_delta must be nonnegative.")
        if self.enabled:
            if self.patience is None or self.patience <= 0:
                raise ValueError(
                    "Enabled early stopping requires a positive patience."
                )
        elif self.patience is not None and self.patience <= 0:
            raise ValueError(
                "Early-stopping patience must be positive when provided."
            )


@dataclass(frozen=True)
class TrainingConfig:
    """One-candidate training configuration for the unified trainer."""

    epochs: int = 200
    batch_size: int = 16
    num_workers: int = 0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    checkpoint_selection: ValidationSelectionRule = field(
        default_factory=ValidationSelectionRule
    )
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )
    determinism: DeterminismConfig = field(
        default_factory=DeterminismConfig
    )

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative.")
        if self.lr_schedule.min_lr > self.optimizer.lr:
            raise ValueError(
                "Scheduler min_lr cannot exceed the optimizer base lr."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        _require_json_serializable(payload)
        return payload


@dataclass(frozen=True)
class RunIdentity:
    """
    Scientific identity of one candidate run.

    ``split_identity`` is deliberately opaque at Stage 6A. Stage 6B will
    define how a stable split fingerprint is produced and validated for
    checkpoint resume. A filesystem path alone is not treated as proof of
    split identity.
    """

    mode: TrainingMode
    seed: int
    split_identity: str
    backbone_config: Mapping[str, Any]
    temporal_config: Mapping[str, Any] | None = None
    window_policy: WindowPolicyConfig | None = None
    fusion_config: Mapping[str, Any] | None = None
    # Fusion only. When None (the historical case) both branches are built
    # from ``backbone_config``. When set, the temporal branch is built from it
    # instead - this is what lets X + TempLasso pair a BQN static branch with
    # a lasso_edges window encoder.
    temporal_backbone_config: Mapping[str, Any] | None = None
    # Output classes of every pipeline head. 2 is the historical binary
    # protocol and is NOT serialised, so every existing candidate keeps its
    # canonical JSON and therefore its fingerprint and resumable paths.
    num_classes: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.num_classes, bool) or not isinstance(self.num_classes, int):
            raise TypeError("num_classes must be an integer.")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        if not isinstance(self.split_identity, str) or not self.split_identity.strip():
            raise ValueError("split_identity must be a non-empty string.")

        if self.mode != "fusion" and self.temporal_backbone_config is not None:
            raise ValueError(
                "temporal_backbone_config is only meaningful for fusion; "
                f"{self.mode} mode has a single backbone_config."
            )

        if self.mode == "vanilla":
            if self.temporal_config is not None:
                raise ValueError("Vanilla mode must not define temporal_config.")
            if self.window_policy is not None:
                raise ValueError("Vanilla mode must not define window_policy.")
            if self.fusion_config is not None:
                raise ValueError("Vanilla mode must not define fusion_config.")
        elif self.mode == "temporal":
            if self.temporal_config is None:
                raise ValueError("Temporal mode requires temporal_config.")
            if self.window_policy is None:
                raise ValueError("Temporal mode requires window_policy.")
            if self.fusion_config is not None:
                raise ValueError("Temporal mode must not define fusion_config.")
        elif self.mode == "fusion":
            if self.temporal_config is None:
                raise ValueError("Fusion mode requires temporal_config.")
            if self.window_policy is None:
                raise ValueError("Fusion mode requires window_policy.")
            if self.fusion_config is None:
                raise ValueError("Fusion mode requires fusion_config.")
        else:
            raise ValueError(
                "Stage 6 supports only mode='vanilla', mode='temporal', "
                "or mode='fusion'."
            )

        _require_json_serializable(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        # Preserve the exact serialized identity of already-validated
        # vanilla/temporal candidates.  In particular, do not inject a new
        # ``fusion_config: null`` field into their canonical JSON because
        # candidate fingerprints and resumable artifact paths depend on this
        # representation.
        payload = {
            "mode": self.mode,
            "seed": self.seed,
            "split_identity": self.split_identity,
            "backbone_config": dict(self.backbone_config),
            "temporal_config": (
                None
                if self.temporal_config is None
                else dict(self.temporal_config)
            ),
            "window_policy": (
                None
                if self.window_policy is None
                else asdict(self.window_policy)
            ),
        }

        if self.mode == "fusion":
            payload["fusion_config"] = dict(self.fusion_config or {})
            # Emitted only when present: historical fusion candidates keep
            # their exact canonical JSON and therefore their fingerprints.
            if self.temporal_backbone_config is not None:
                payload["temporal_backbone_config"] = dict(
                    self.temporal_backbone_config
                )

        if self.num_classes != 2:
            payload["num_classes"] = int(self.num_classes)

        return payload


@dataclass(frozen=True)
class CandidateRunConfig:
    """Complete serializable contract for one candidate training run."""

    identity: RunIdentity
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "identity": self.identity.to_dict(),
            "training": self.training.to_dict(),
        }
        _require_json_serializable(payload)
        return payload


def _require_json_serializable(payload: Mapping[str, Any]) -> None:
    try:
        json.dumps(payload, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Run/training configuration must be finite and JSON serializable."
        ) from exc
