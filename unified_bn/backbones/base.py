from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from unified_bn.core.types import BackboneOutput
from unified_bn.core.validation import validate_fc_batch


class BaseBackbone(nn.Module, ABC):
    """Abstract interface for any FC-matrix backbone: [B,R,R] -> [B,D]."""

    def __init__(self, *, n_rois: int, embedding_dim: int) -> None:
        super().__init__()
        if n_rois <= 0:
            raise ValueError("n_rois must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self._n_rois = int(n_rois)
        self._embedding_dim = int(embedding_dim)

    @property
    def n_rois(self) -> int:
        return self._n_rois

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, fc: torch.Tensor) -> BackboneOutput:
        validate_fc_batch(fc, n_rois=self.n_rois)
        output = self._encode_impl(fc)
        if not isinstance(output, BackboneOutput):
            raise TypeError(f"{type(self).__name__}._encode_impl() must return BackboneOutput.")
        embedding = output.embedding
        if embedding.ndim != 2:
            raise ValueError(f"{type(self).__name__} returned embedding shape {tuple(embedding.shape)}; expected [B,D].")
        if embedding.shape[0] != fc.shape[0]:
            raise ValueError(f"{type(self).__name__} changed batch size from {fc.shape[0]} to {embedding.shape[0]}.")
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"{type(self).__name__} declared embedding_dim={self.embedding_dim} but returned D={embedding.shape[1]}.")
        if not torch.isfinite(embedding).all():
            raise ValueError(f"{type(self).__name__} produced NaN or Inf embeddings.")
        return output

    def encode(self, fc: torch.Tensor) -> torch.Tensor:
        return self.forward(fc).embedding

    @abstractmethod
    def _encode_impl(self, fc: torch.Tensor) -> BackboneOutput:
        raise NotImplementedError
