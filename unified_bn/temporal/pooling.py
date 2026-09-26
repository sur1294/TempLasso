from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PoolingOutput:
    """
    Output of a temporal pooling operator.

    Attributes
    ----------
    embedding:
        Subject-level representation with shape [B, D_out].
    weights:
        Optional normalized temporal weights with shape [B, K].
        Present for attention pooling and absent otherwise.
    """

    embedding: torch.Tensor
    weights: Optional[torch.Tensor] = None


def lengths_to_mask(
    lengths: torch.Tensor,
    *,
    max_steps: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert sequence lengths [B] into a boolean validity mask [B, K].
    """
    if lengths.ndim != 1:
        raise ValueError(
            f"lengths must have shape [B], got {tuple(lengths.shape)}."
        )
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    target_device = device if device is not None else lengths.device
    steps = torch.arange(max_steps, device=target_device)
    return steps.unsqueeze(0) < lengths.to(target_device).unsqueeze(1)


class BaseTemporalPooler(nn.Module, ABC):
    """
    Interface for pooling a sequence representation [B,K,D] to [B,D_out].
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        *,
        final_hidden: Optional[torch.Tensor] = None,
    ) -> PoolingOutput:
        raise NotImplementedError


class FinalStatePooling(BaseTemporalPooler):
    """
    Return the top-layer bidirectional recurrent final hidden state.

    `final_hidden` must already have shape [B, D]. This keeps the recurrent
    state interpretation explicit and avoids approximating final states from
    padded sequence outputs.
    """

    def __init__(self, *, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self._output_dim = int(output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        *,
        final_hidden: Optional[torch.Tensor] = None,
    ) -> PoolingOutput:
        del sequence, lengths

        if final_hidden is None:
            raise ValueError(
                "FinalStatePooling requires final_hidden from the recurrent "
                "encoder."
            )
        if final_hidden.ndim != 2:
            raise ValueError(
                "final_hidden must have shape [B,D], got "
                f"{tuple(final_hidden.shape)}."
            )
        if final_hidden.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected final_hidden D={self.output_dim}, got "
                f"{final_hidden.shape[1]}."
            )

        return PoolingOutput(embedding=final_hidden)


class MaskedMeanPooling(BaseTemporalPooler):
    """
    Mean-pool only valid temporal steps.

    This is a deterministic non-parametric pooling baseline.
    """

    def __init__(self, *, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self._input_dim = int(input_dim)

    @property
    def output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        *,
        final_hidden: Optional[torch.Tensor] = None,
    ) -> PoolingOutput:
        del final_hidden

        if sequence.ndim != 3:
            raise ValueError(
                f"sequence must have shape [B,K,D], got {tuple(sequence.shape)}."
            )
        if sequence.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected D={self.output_dim}, got {sequence.shape[-1]}."
            )

        mask = lengths_to_mask(
            lengths,
            max_steps=sequence.shape[1],
            device=sequence.device,
        )
        masked = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        denom = lengths.to(sequence.device, dtype=sequence.dtype).unsqueeze(-1)
        embedding = masked.sum(dim=1) / denom

        return PoolingOutput(embedding=embedding)


class MaskedAttentionPooling(BaseTemporalPooler):
    """
    Learned scalar attention over valid temporal steps.

    Scores are produced independently for each time step:
        score_t = w2(tanh(W1 h_t))

    Invalid/padded steps are assigned -inf before softmax, guaranteeing zero
    attention probability on padding.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self._input_dim = int(input_dim)

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    @property
    def output_dim(self) -> int:
        return self._input_dim

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        *,
        final_hidden: Optional[torch.Tensor] = None,
    ) -> PoolingOutput:
        del final_hidden

        if sequence.ndim != 3:
            raise ValueError(
                f"sequence must have shape [B,K,D], got {tuple(sequence.shape)}."
            )
        if sequence.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected D={self.output_dim}, got {sequence.shape[-1]}."
            )

        mask = lengths_to_mask(
            lengths,
            max_steps=sequence.shape[1],
            device=sequence.device,
        )

        scores = self.scorer(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)

        # Defensive invariant: padded steps must have exactly zero probability.
        weights = weights.masked_fill(~mask, 0.0)

        embedding = torch.bmm(
            weights.unsqueeze(1),
            sequence,
        ).squeeze(1)

        return PoolingOutput(
            embedding=embedding,
            weights=weights,
        )
