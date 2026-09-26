from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from unified_bn.core.types import TemporalOutput
from unified_bn.core.validation import validate_lengths, validate_sequence_embeddings


class BaseTemporalAggregator(nn.Module, ABC):
    """Abstract interface for [B,K,D_in] + lengths -> [B,D_out]."""

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        self._input_dim = int(input_dim)
        self._output_dim = int(output_dim)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> TemporalOutput:
        validate_sequence_embeddings(sequence, embedding_dim=self.input_dim)
        validate_lengths(lengths, batch_size=sequence.shape[0], max_steps=sequence.shape[1])
        output = self._aggregate_impl(sequence, lengths)
        if not isinstance(output, TemporalOutput):
            raise TypeError(f"{type(self).__name__}._aggregate_impl() must return TemporalOutput.")
        embedding = output.embedding
        expected = (sequence.shape[0], self.output_dim)
        if embedding.shape != expected:
            raise ValueError(f"{type(self).__name__} expected output shape {expected}, got {tuple(embedding.shape)}.")
        if not torch.isfinite(embedding).all():
            raise ValueError(f"{type(self).__name__} produced NaN or Inf embeddings.")
        return output

    @abstractmethod
    def _aggregate_impl(self, sequence: torch.Tensor, lengths: torch.Tensor) -> TemporalOutput:
        raise NotImplementedError
