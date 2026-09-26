from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch.optim import Optimizer

from unified_bn.training.config import LRScheduleConfig, OptimizerConfig


class OptimizationConfigError(ValueError):
    """Raised when a configured optimizer/scheduler cannot be constructed safely."""


class EpochLRScheduler:
    """
    Epoch-indexed LR controller matching the validated historical BQN protocol.

    The current Atlas ABIDE default is cosine decay from each optimizer
    parameter group's configured base LR toward ``schedule_config.min_lr``.
    Crucially, no learning rate or horizon is hard-coded here: base LRs come
    from the optimizer configuration and the horizon comes from the candidate's
    configured epoch budget.

    This object intentionally exposes ``state_dict`` / ``load_state_dict`` so
    Stage 6B checkpointing can restore scheduler state without coupling the
    trainer to a specific PyTorch scheduler implementation.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        schedule_config: LRScheduleConfig,
        total_epochs: int,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "optimizer must be torch.optim.Optimizer, got "
                f"{type(optimizer)!r}."
            )
        if not isinstance(schedule_config, LRScheduleConfig):
            raise TypeError(
                "schedule_config must be LRScheduleConfig, got "
                f"{type(schedule_config)!r}."
            )
        if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
            raise TypeError("total_epochs must be an integer.")
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive.")
        if not optimizer.param_groups:
            raise OptimizationConfigError("Optimizer has no parameter groups.")

        self.optimizer = optimizer
        self.schedule_config = schedule_config
        self.total_epochs = int(total_epochs)
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        self.last_epoch_index = -1

        for lr in self.base_lrs:
            if not math.isfinite(lr) or lr <= 0.0:
                raise OptimizationConfigError(
                    "All optimizer parameter-group learning rates must be finite "
                    "and positive."
                )
            if schedule_config.min_lr > lr:
                raise OptimizationConfigError(
                    "Scheduler min_lr cannot exceed a parameter group's base LR: "
                    f"min_lr={schedule_config.min_lr}, base_lr={lr}."
                )

    def set_epoch(self, epoch_index: int) -> tuple[float, ...]:
        """
        Set LR at the beginning of a zero-based training epoch.

        For the current cosine protocol, epoch 0 uses the configured base LR
        and the final configured epoch uses exactly ``min_lr``. ``constant``
        leaves all parameter-group LRs at their configured base values.
        """

        if isinstance(epoch_index, bool) or not isinstance(epoch_index, int):
            raise TypeError("epoch_index must be an integer.")
        if epoch_index < 0 or epoch_index >= self.total_epochs:
            raise ValueError(
                "epoch_index must satisfy 0 <= epoch_index < total_epochs; "
                f"got epoch_index={epoch_index}, total_epochs={self.total_epochs}."
            )

        lrs = tuple(self._lr_for_epoch(base_lr, epoch_index) for base_lr in self.base_lrs)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = float(lr)

        self.last_epoch_index = int(epoch_index)
        return lrs

    def get_last_lr(self) -> tuple[float, ...]:
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self._STATE_VERSION,
            "schedule_name": self.schedule_config.name,
            "min_lr": float(self.schedule_config.min_lr),
            "total_epochs": int(self.total_epochs),
            "base_lrs": list(self.base_lrs),
            "last_epoch_index": int(self.last_epoch_index),
            "current_lrs": list(self.get_last_lr()),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("scheduler state_dict must be a mapping.")

        required = {
            "state_version",
            "schedule_name",
            "min_lr",
            "total_epochs",
            "base_lrs",
            "last_epoch_index",
            "current_lrs",
        }
        missing = required - set(state_dict)
        if missing:
            raise OptimizationConfigError(
                f"Scheduler state is missing fields: {sorted(missing)}."
            )

        if int(state_dict["state_version"]) != self._STATE_VERSION:
            raise OptimizationConfigError(
                "Unsupported EpochLRScheduler state version: "
                f"{state_dict['state_version']!r}."
            )
        if state_dict["schedule_name"] != self.schedule_config.name:
            raise OptimizationConfigError(
                "Scheduler name differs from configured run."
            )
        if not math.isclose(
            float(state_dict["min_lr"]),
            float(self.schedule_config.min_lr),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise OptimizationConfigError(
                "Scheduler min_lr differs from configured run."
            )
        if int(state_dict["total_epochs"]) != self.total_epochs:
            raise OptimizationConfigError(
                "Scheduler total_epochs differs from configured run."
            )

        restored_base = tuple(float(x) for x in state_dict["base_lrs"])
        restored_current = tuple(float(x) for x in state_dict["current_lrs"])
        if restored_base != self.base_lrs:
            raise OptimizationConfigError(
                "Scheduler base learning rates differ from configured optimizer."
            )
        if len(restored_current) != len(self.optimizer.param_groups):
            raise OptimizationConfigError(
                "Scheduler current LR count differs from optimizer parameter groups."
            )

        last_epoch_index = int(state_dict["last_epoch_index"])
        if last_epoch_index < -1 or last_epoch_index >= self.total_epochs:
            raise OptimizationConfigError(
                "Scheduler last_epoch_index is outside the configured horizon."
            )

        for lr in restored_current:
            if not math.isfinite(lr) or lr < 0.0:
                raise OptimizationConfigError(
                    "Scheduler current learning rates must be finite and nonnegative."
                )

        for group, lr in zip(self.optimizer.param_groups, restored_current):
            group["lr"] = float(lr)
        self.last_epoch_index = last_epoch_index

    def _lr_for_epoch(self, base_lr: float, epoch_index: int) -> float:
        name = self.schedule_config.name
        if name == "constant":
            return float(base_lr)
        if name == "cosine":
            if self.total_epochs <= 1:
                # Match the validated reference BQN implementation exactly.
                return float(self.schedule_config.min_lr)
            progress = epoch_index / (self.total_epochs - 1)
            return float(
                self.schedule_config.min_lr
                + 0.5
                * (base_lr - self.schedule_config.min_lr)
                * (1.0 + math.cos(math.pi * progress))
            )
        raise OptimizationConfigError(
            f"Unsupported LR schedule: {name!r}."
        )


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> Optimizer:
    """
    Build the optimizer for one candidate from configuration only.

    The current protocol supports Adam because that is the verified Atlas/BQN
    setting. LR and weight decay are never hard-coded here; changing them for
    a future tuning grid requires only a different ``OptimizerConfig``.
    Additional optimizer families can be added to this factory without
    modifying the generic trainer.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)!r}.")
    if not isinstance(config, OptimizerConfig):
        raise TypeError(
            "config must be OptimizerConfig, got "
            f"{type(config)!r}."
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise OptimizationConfigError("Model has no trainable parameters.")

    if config.name == "adam":
        return torch.optim.Adam(
            trainable_parameters,
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

    raise OptimizationConfigError(
        f"Unsupported optimizer family: {config.name!r}."
    )


def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    schedule_config: LRScheduleConfig,
    total_epochs: int,
) -> EpochLRScheduler:
    """Build an epoch-indexed LR schedule from candidate configuration."""

    return EpochLRScheduler(
        optimizer,
        schedule_config=schedule_config,
        total_epochs=total_epochs,
    )
