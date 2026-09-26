#!/usr/bin/env python3
"""
Bidirectional GRU temporal aggregator for BQN+.

SCOPE
-----
This file contains ONLY the temporal sequence model.

Input
-----
window_embeddings : float tensor [B, Kmax, D_in]
    Per-window embeddings produced by the SHARED BQN backbone.

lengths : long tensor [B]
    Number of valid temporal windows for each subject.

Optional:
valid_mask : bool tensor [B, Kmax]
    True for valid windows, False for padding.
    If supplied, it is checked against lengths.

Output
------
dynamic_embedding : [B, D_out]

Optional auxiliary outputs:
    sequence_output   [B, Kmax, 2*hidden_dim]
    attention_weights [B, Kmax]
    valid_mask        [B, Kmax]

DESIGN
------
1. Variable-length handling:
   Uses pack_padded_sequence so the recurrent state never processes padded
   windows as if they were real observations.

2. Bidirectionality:
   Appropriate for offline whole-scan subject classification, where the full
   sequence is available and no causal forecasting constraint exists.

3. Temporal pooling:
   Uses learned masked attention pooling over all valid BiGRU outputs instead
   of only the final hidden state.

4. Projection:
   Projects the pooled 2H-dimensional BiGRU representation to D_out so the
   dynamic representation can later be fused cleanly with the static BQN
   embedding.

5. No classifier and no fusion are included here.
   Those belong in the later BQN+ model.

Recommended initial setting for this project:
    input_dim = 32
    hidden_dim = 32
    num_layers = 1
    output_dim = 32

Research tuning can later consider:
    hidden_dim in {32, 64}
    num_layers in {1, 2}
    dropout in {0.0, 0.1, 0.2, 0.3}

IMPORTANT
---------
For torch.nn.GRU, recurrent dropout is only active when num_layers > 1.
This module therefore automatically sets GRU dropout to 0.0 when
num_layers == 1, even if a nonzero dropout value is requested. The requested
dropout can still be used in the output projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


@dataclass(frozen=True)
class BiGRUConfig:
    input_dim: int = 32
    hidden_dim: int = 32
    num_layers: int = 1
    output_dim: int = 32
    dropout: float = 0.1
    attention_hidden_dim: int = 32
    bidirectional: bool = True


class MaskedAttentionPooling(nn.Module):
    """
    Learned additive attention pooling over valid temporal positions.

    For sequence states h_t:

        e_t = v^T tanh(W h_t + b)

    Invalid/padded positions are masked to -inf before softmax.

        alpha_t = softmax(e_t)

    and the pooled representation is:

        z = sum_t alpha_t h_t

    Input:
        x          [B,K,D]
        valid_mask [B,K] bool

    Output:
        pooled     [B,D]
        weights    [B,K]
    """

    def __init__(
        self,
        input_dim: int,
        attention_hidden_dim: int,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be positive.")

        self.input_dim = int(input_dim)
        self.attention_hidden_dim = int(attention_hidden_dim)

        self.score_net = nn.Sequential(
            nn.Linear(self.input_dim, self.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.attention_hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B,K,D], got {tuple(x.shape)}"
            )

        if valid_mask.ndim != 2:
            raise ValueError(
                f"valid_mask must have shape [B,K], got "
                f"{tuple(valid_mask.shape)}"
            )

        if x.shape[:2] != valid_mask.shape:
            raise ValueError(
                f"x and valid_mask disagree on [B,K]: "
                f"{tuple(x.shape[:2])} vs {tuple(valid_mask.shape)}"
            )

        if valid_mask.dtype != torch.bool:
            raise TypeError(
                f"valid_mask must be bool, got {valid_mask.dtype}"
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dim {self.input_dim}, got {x.shape[-1]}"
            )

        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError(
                "Every subject must have at least one valid temporal window."
            )

        scores = self.score_net(x).squeeze(-1)  # [B,K]

        # Use a finite minimum representable value rather than literal -inf
        # to avoid NaN propagation on some mixed-precision paths.
        neg_large = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid_mask, neg_large)

        weights = torch.softmax(scores, dim=1)

        # Explicitly zero masked entries, then renormalize over valid entries.
        weights = weights * valid_mask.to(weights.dtype)
        denom = weights.sum(dim=1, keepdim=True)

        if torch.any(denom <= 0):
            raise RuntimeError(
                "Attention normalization failed: a subject has zero valid mass."
            )

        weights = weights / denom

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled, weights


class BiGRUTemporalAggregator(nn.Module):
    """
    Bidirectional GRU + masked attention pooling.

    Input:
        window_embeddings [B,Kmax,input_dim]
        lengths           [B]
        valid_mask        optional [B,Kmax]

    Output:
        dynamic_embedding [B,output_dim]

    If return_aux=True:
        returns (dynamic_embedding, aux_dict)
    """

    def __init__(
        self,
        config: Optional[BiGRUConfig] = None,
    ) -> None:
        super().__init__()

        self.config = (
            config
            if config is not None
            else BiGRUConfig()
        )

        self._validate_config()

        directions = 2 if self.config.bidirectional else 1
        gru_output_dim = self.config.hidden_dim * directions

        # PyTorch GRU recurrent dropout only applies BETWEEN stacked layers.
        gru_dropout = (
            self.config.dropout
            if self.config.num_layers > 1
            else 0.0
        )

        self.gru = nn.GRU(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=self.config.bidirectional,
        )

        self.attention_pool = MaskedAttentionPooling(
            input_dim=gru_output_dim,
            attention_hidden_dim=self.config.attention_hidden_dim,
        )

        # Keep this lightweight.
        self.output_projection = nn.Sequential(
            nn.Linear(gru_output_dim, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        self.sequence_dim = gru_output_dim
        self.output_dim = self.config.output_dim

    def _validate_config(self) -> None:
        c = self.config

        if c.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if c.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if c.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if c.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if c.attention_hidden_dim <= 0:
            raise ValueError(
                "attention_hidden_dim must be positive."
            )
        if not 0.0 <= c.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

    @staticmethod
    def lengths_to_mask(
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """
        Convert lengths [B] to bool valid-mask [B,max_len].
        """
        if lengths.ndim != 1:
            raise ValueError(
                f"lengths must be [B], got {tuple(lengths.shape)}"
            )

        if lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                f"lengths must be integer tensor, got {lengths.dtype}"
            )

        if torch.any(lengths <= 0):
            raise ValueError("All sequence lengths must be positive.")

        if torch.any(lengths > max_len):
            raise ValueError(
                "A sequence length exceeds the padded K dimension."
            )

        positions = torch.arange(
            max_len,
            device=lengths.device,
        ).unsqueeze(0)

        return positions < lengths.unsqueeze(1)

    def _validate_inputs(
        self,
        window_embeddings: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if window_embeddings.ndim != 3:
            raise ValueError(
                "window_embeddings must have shape [B,K,D], got "
                f"{tuple(window_embeddings.shape)}"
            )

        b, kmax, d = window_embeddings.shape

        if d != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {d}"
            )

        if lengths.ndim != 1 or lengths.shape[0] != b:
            raise ValueError(
                f"lengths must have shape [{b}], got {tuple(lengths.shape)}"
            )

        if not torch.is_floating_point(window_embeddings):
            raise TypeError(
                "window_embeddings must be floating-point."
            )

        if not torch.isfinite(window_embeddings).all():
            raise ValueError(
                "window_embeddings contain NaN or Inf."
            )

        derived_mask = self.lengths_to_mask(
            lengths=lengths,
            max_len=kmax,
        )

        if valid_mask is None:
            return derived_mask

        if valid_mask.shape != (b, kmax):
            raise ValueError(
                f"valid_mask must have shape {(b, kmax)}, "
                f"got {tuple(valid_mask.shape)}"
            )

        if valid_mask.dtype != torch.bool:
            raise TypeError(
                f"valid_mask must be bool, got {valid_mask.dtype}"
            )

        if not torch.equal(valid_mask, derived_mask):
            raise ValueError(
                "valid_mask does not exactly match lengths."
            )

        return valid_mask

    def forward(
        self,
        window_embeddings: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        Run BiGRU on variable-length temporal BQN embeddings.

        `lengths` may live on GPU, but pack_padded_sequence requires a CPU
        lengths tensor, so a detached CPU copy is used only for packing.
        """
        valid_mask = self._validate_inputs(
            window_embeddings=window_embeddings,
            lengths=lengths,
            valid_mask=valid_mask,
        )

        _, kmax, _ = window_embeddings.shape

        lengths_cpu = (
            lengths.detach()
            .to(device="cpu", dtype=torch.long)
        )

        packed = pack_padded_sequence(
            window_embeddings,
            lengths=lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, hidden = self.gru(packed)

        sequence_output, unpacked_lengths = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=kmax,
        )

        # Strong consistency check.
        unpacked_lengths = unpacked_lengths.to(
            device=lengths.device,
            dtype=lengths.dtype,
        )

        if not torch.equal(unpacked_lengths, lengths):
            raise RuntimeError(
                "pad_packed_sequence lengths do not match input lengths."
            )

        pooled, attention_weights = self.attention_pool(
            x=sequence_output,
            valid_mask=valid_mask,
        )

        dynamic_embedding = self.output_projection(pooled)

        if not torch.isfinite(dynamic_embedding).all():
            raise FloatingPointError(
                "BiGRU dynamic embedding contains NaN or Inf."
            )

        if return_aux:
            aux: Dict[str, torch.Tensor] = {
                "sequence_output": sequence_output,
                "attention_weights": attention_weights,
                "valid_mask": valid_mask,
                "hidden": hidden,
            }
            return dynamic_embedding, aux

        return dynamic_embedding

    def parameter_summary(self) -> Dict[str, int]:
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(
                sum(
                    p.numel()
                    for p in self.parameters()
                    if p.requires_grad
                )
            ),
            "gru": int(
                sum(p.numel() for p in self.gru.parameters())
            ),
            "attention_pool": int(
                sum(
                    p.numel()
                    for p in self.attention_pool.parameters()
                )
            ),
            "output_projection": int(
                sum(
                    p.numel()
                    for p in self.output_projection.parameters()
                )
            ),
        }


def _self_test() -> None:
    torch.manual_seed(42)

    config = BiGRUConfig(
        input_dim=32,
        hidden_dim=32,
        num_layers=1,
        output_dim=32,
        dropout=0.1,
        attention_hidden_dim=32,
        bidirectional=True,
    )

    model = BiGRUTemporalAggregator(config)
    model.train()

    # Mimic one batch from temporal_dataset.py.
    lengths = torch.tensor(
        [15, 26, 17, 13],
        dtype=torch.long,
    )
    kmax = int(lengths.max().item())

    x = torch.randn(
        4,
        kmax,
        32,
        dtype=torch.float32,
    )

    valid_mask = model.lengths_to_mask(
        lengths,
        max_len=kmax,
    )

    # Make padded embeddings nonzero on purpose.
    # pack_padded_sequence must ensure they do not enter the recurrent
    # computation, and masked pooling must ensure they receive no attention.
    x = x.clone()
    x[~valid_mask] = 123.456

    z, aux = model(
        window_embeddings=x,
        lengths=lengths,
        valid_mask=valid_mask,
        return_aux=True,
    )

    assert z.shape == (4, 32)
    assert aux["sequence_output"].shape == (4, 26, 64)
    assert aux["attention_weights"].shape == (4, 26)
    assert aux["valid_mask"].shape == (4, 26)

    # Attention on padding must be exactly zero.
    assert torch.all(
        aux["attention_weights"][~valid_mask] == 0
    )

    # Attention over valid windows must sum to one per subject.
    weight_sum = aux["attention_weights"].sum(dim=1)
    assert torch.allclose(
        weight_sum,
        torch.ones_like(weight_sum),
        atol=1e-6,
    )

    # Verify gradients through the full temporal aggregator.
    loss = z.square().mean()
    loss.backward()

    grad_checks = {
        "gru": any(
            p.grad is not None
            for p in model.gru.parameters()
            if p.requires_grad
        ),
        "attention": any(
            p.grad is not None
            for p in model.attention_pool.parameters()
            if p.requires_grad
        ),
        "projection": any(
            p.grad is not None
            for p in model.output_projection.parameters()
            if p.requires_grad
        ),
    }

    assert all(grad_checks.values())

    print("BiGRUTemporalAggregator self-test: PASS")
    print("dynamic embedding:", tuple(z.shape))
    print(
        "sequence output:",
        tuple(aux["sequence_output"].shape),
    )
    print(
        "attention weights:",
        tuple(aux["attention_weights"].shape),
    )
    print(
        "attention sums:",
        aux["attention_weights"].sum(dim=1).detach().tolist(),
    )
    print(
        "padding attention exactly zero:",
        bool(
            torch.all(
                aux["attention_weights"][~valid_mask] == 0
            )
        ),
    )
    print("gradient checks:", grad_checks)
    print("parameter summary:", model.parameter_summary())


if __name__ == "__main__":
    _self_test()
