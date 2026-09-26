from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class BackboneOutput:
    """Standard output of an FC-matrix backbone."""
    embedding: torch.Tensor
    aux: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalOutput:
    """Standard output of a temporal sequence aggregator."""
    embedding: torch.Tensor
    aux: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionOutput:
    """Standard output of a static/temporal fusion module."""
    embedding: torch.Tensor
    aux: Mapping[str, Any] = field(default_factory=dict)
