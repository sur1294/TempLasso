from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

WindowPolicy = Literal["native", "min"]
WindowCrop = Literal["start", "center"]


@dataclass(frozen=True)
class WindowPolicyConfig:
    """
    Temporal-window policy applied before backbone encoding.

    policy='native'
        Preserve every subject's native number of windows.

    policy='min'
        Truncate every subject deterministically to a precomputed common
        window count `min_windows`.

    IMPORTANT
    ---------
    `min_windows` must be resolved from the intended dataset cohort outside
    the minibatch loop. It must never be recomputed independently per batch.
    """

    policy: WindowPolicy = "native"
    crop: WindowCrop = "start"
    min_windows: int | None = None


def resolve_global_min_windows(lengths: torch.Tensor) -> int:
    """
    Resolve the minimum window count from a complete dataset/cohort length
    vector. This helper is intentionally separate from minibatch processing.
    """
    if lengths.ndim != 1 or lengths.numel() == 0:
        raise ValueError("lengths must be a non-empty 1D tensor.")
    if torch.any(lengths < 1):
        raise ValueError("All lengths must be positive.")
    return int(lengths.min().item())


def apply_window_policy(
    dynamic_fc: torch.Tensor,
    lengths: torch.Tensor,
    config: WindowPolicyConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply native or deterministic common-minimum window handling.

    Inputs
    ------
    dynamic_fc:
        [B,K,R,R]
    lengths:
        [B]

    Returns
    -------
    dynamic_fc_out:
        [B,K_out,R,R]
    lengths_out:
        [B]
    """
    if dynamic_fc.ndim != 4:
        raise ValueError(
            "dynamic_fc must have shape [B,K,R,R], got "
            f"{tuple(dynamic_fc.shape)}."
        )
    if lengths.ndim != 1 or lengths.shape[0] != dynamic_fc.shape[0]:
        raise ValueError(
            "lengths must have shape [B] matching dynamic_fc batch size."
        )

    if config.policy == "native":
        return dynamic_fc, lengths

    if config.policy != "min":
        raise ValueError(f"Unsupported window policy: {config.policy!r}")

    if config.min_windows is None:
        raise ValueError(
            "policy='min' requires a precomputed min_windows value."
        )

    target = int(config.min_windows)

    if target <= 0:
        raise ValueError("min_windows must be positive.")

    if torch.any(lengths < target):
        raise ValueError(
            "At least one subject has fewer windows than min_windows="
            f"{target}."
        )

    if target > dynamic_fc.shape[1]:
        raise ValueError(
            f"min_windows={target} exceeds padded K={dynamic_fc.shape[1]}."
        )

    if config.crop == "start":
        out = dynamic_fc[:, :target]
    elif config.crop == "center":
        rows = []
        for i in range(dynamic_fc.shape[0]):
            length = int(lengths[i].item())
            start = (length - target) // 2
            rows.append(dynamic_fc[i, start:start + target])
        out = torch.stack(rows, dim=0)
    else:
        raise ValueError(f"Unsupported crop mode: {config.crop!r}")

    new_lengths = torch.full_like(lengths, target)
    return out, new_lengths
