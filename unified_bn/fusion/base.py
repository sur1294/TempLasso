from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from unified_bn.core.types import FusionOutput


class BaseFusion(nn.Module, ABC):
    """Abstract interface for [B,D_s] + [B,D_t] -> [B,D_f]."""

    def __init__(self, *, static_dim: int, temporal_dim: int, output_dim: int) -> None:
        super().__init__()
        if static_dim <= 0 or temporal_dim <= 0 or output_dim <= 0:
            raise ValueError("static_dim, temporal_dim, and output_dim must be positive.")
        self._static_dim = int(static_dim)
        self._temporal_dim = int(temporal_dim)
        self._output_dim = int(output_dim)

    @property
    def static_dim(self) -> int:
        return self._static_dim

    @property
    def temporal_dim(self) -> int:
        return self._temporal_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, h_static: torch.Tensor, h_temporal: torch.Tensor) -> FusionOutput:
        self._validate_inputs(h_static, h_temporal)
        output = self._fuse_impl(h_static, h_temporal)
        if not isinstance(output, FusionOutput):
            raise TypeError(f"{type(self).__name__}._fuse_impl() must return FusionOutput.")
        expected = (h_static.shape[0], self.output_dim)
        if output.embedding.shape != expected:
            raise ValueError(f"{type(self).__name__} expected fused shape {expected}, got {tuple(output.embedding.shape)}.")
        if not torch.isfinite(output.embedding).all():
            raise ValueError(f"{type(self).__name__} produced NaN or Inf embeddings.")
        return output

    def _validate_inputs(self, h_static: torch.Tensor, h_temporal: torch.Tensor) -> None:
        for name, tensor, dim in (("h_static", h_static, self.static_dim), ("h_temporal", h_temporal, self.temporal_dim)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}.")
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape [B,D], got {tuple(tensor.shape)}.")
            if tensor.shape[1] != dim:
                raise ValueError(f"{name} expected D={dim}, got D={tensor.shape[1]}.")
            if not torch.is_floating_point(tensor):
                raise TypeError(f"{name} must be floating point, got {tensor.dtype}.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf.")
        if h_static.shape[0] != h_temporal.shape[0]:
            raise ValueError(f"Static and temporal embeddings must have identical batch size; got {h_static.shape[0]} vs {h_temporal.shape[0]}.")

    @abstractmethod
    def _fuse_impl(self, h_static: torch.Tensor, h_temporal: torch.Tensor) -> FusionOutput:
        raise NotImplementedError
