from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from unified_bn.core.types import TemporalOutput
from unified_bn.temporal.base import BaseTemporalAggregator
from unified_bn.temporal.pooling import (
    FinalStatePooling,
    MaskedAttentionPooling,
    MaskedMeanPooling,
)

PoolingMode = Literal["final", "mean", "attention"]


@dataclass(frozen=True)
class BiGRUConfig:
    """
    Configuration for the bidirectional GRU temporal encoder.

    Parameters
    ----------
    input_dim:
        Backbone window-embedding dimension.
    hidden_dim:
        Hidden size per GRU direction.
    num_layers:
        Number of stacked recurrent layers.
    dropout:
        Inter-layer GRU dropout. PyTorch applies recurrent-stack dropout only
        when num_layers > 1.
    pooling:
        Subject-level pooling strategy: final, mean, or attention.
    attention_hidden_dim:
        Hidden width of the attention scorer when pooling='attention'.
    attention_dropout:
        Dropout inside the attention scorer.
    projection_dim:
        Optional output projection dimension after pooling.
    layer_norm:
        Apply LayerNorm after optional projection.
    """

    input_dim: int
    hidden_dim: int
    num_layers: int = 1
    dropout: float = 0.0
    pooling: PoolingMode = "final"
    attention_hidden_dim: int = 32
    attention_dropout: float = 0.0
    projection_dim: Optional[int] = None
    layer_norm: bool = False


class BiGRUAggregator(BaseTemporalAggregator):
    """
    Production bidirectional-GRU temporal aggregator.

    Input
    -----
    sequence : [B,K,D_in]
        Window embeddings in temporal order.
    lengths : [B]
        Number of valid windows for each subject.

    Processing
    ----------
    1. Pack variable-length sequences so padded steps are never processed.
    2. Encode with a stacked bidirectional GRU.
    3. Reconstruct padded sequence outputs only for pooling/diagnostics.
    4. Pool with one explicit strategy.
    5. Optionally project and normalize the subject embedding.

    Padding safety
    --------------
    Padded input values cannot influence the recurrent state because
    `pack_padded_sequence` excludes them from GRU computation.
    """

    def __init__(self, config: BiGRUConfig) -> None:
        self.config = config
        self._validate_config(config)

        recurrent_dim = 2 * config.hidden_dim

        output_dim = (
            config.projection_dim
            if config.projection_dim is not None
            else recurrent_dim
        )

        super().__init__(
            input_dim=config.input_dim,
            output_dim=output_dim,
        )

        gru_dropout = config.dropout if config.num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )

        if config.pooling == "final":
            self.pooler = FinalStatePooling(output_dim=recurrent_dim)
        elif config.pooling == "mean":
            self.pooler = MaskedMeanPooling(input_dim=recurrent_dim)
        elif config.pooling == "attention":
            self.pooler = MaskedAttentionPooling(
                input_dim=recurrent_dim,
                hidden_dim=config.attention_hidden_dim,
                dropout=config.attention_dropout,
            )
        else:
            raise ValueError(f"Unsupported pooling mode: {config.pooling!r}")

        if config.projection_dim is None:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(
                recurrent_dim,
                config.projection_dim,
            )

        if config.layer_norm:
            self.normalization = nn.LayerNorm(output_dim)
        else:
            self.normalization = nn.Identity()

    @staticmethod
    def _validate_config(config: BiGRUConfig) -> None:
        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        if config.pooling not in {"final", "mean", "attention"}:
            raise ValueError(
                "pooling must be one of {'final', 'mean', 'attention'}."
            )
        if config.attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be positive.")
        if not 0.0 <= config.attention_dropout < 1.0:
            raise ValueError(
                "attention_dropout must satisfy 0 <= dropout < 1."
            )
        if (
            config.projection_dim is not None
            and config.projection_dim <= 0
        ):
            raise ValueError("projection_dim must be positive when provided.")

    @property
    def recurrent_dim(self) -> int:
        return 2 * self.config.hidden_dim

    def _aggregate_impl(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
    ) -> TemporalOutput:
        # pack_padded_sequence currently requires CPU lengths.
        packed = pack_padded_sequence(
            sequence,
            lengths.detach().to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, h_n = self.gru(packed)

        recurrent_sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence.shape[1],
        )

        # h_n shape:
        # [num_layers * num_directions, B, hidden_dim]
        # Last two entries are top-layer forward and backward states.
        forward_final = h_n[-2]
        backward_final = h_n[-1]
        final_hidden = torch.cat(
            [forward_final, backward_final],
            dim=-1,
        )

        pooled = self.pooler(
            recurrent_sequence,
            lengths,
            final_hidden=final_hidden,
        )

        embedding = self.projection(pooled.embedding)
        embedding = self.normalization(embedding)

        aux = {
            "sequence_output": recurrent_sequence,
            "final_hidden": final_hidden,
        }

        if pooled.weights is not None:
            aux["attention_weights"] = pooled.weights

        return TemporalOutput(
            embedding=embedding,
            aux=aux,
        )
