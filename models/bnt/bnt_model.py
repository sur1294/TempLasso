"""The user's ALTER/models/transformer/GraphTransformer, exposed as BNT.

This is the supplied plain GraphTransformer, not models/lrbgt's ALTER model.
See UPSTREAM.md for exact source and interface changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class BNTConfig:
    n_rois: int = 100
    n_layers: int = 2
    readout: str = "mean"
    nhead: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    num_classes: int = 2

    def __post_init__(self):
        for name in ("n_rois", "n_layers", "nhead", "ffn_dim", "num_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        if self.n_rois % self.nhead:
            raise ValueError("The FC feature width n_rois must be divisible by nhead.")
        if self.readout not in ("mean", "concat", "max", "sum"):
            raise ValueError("readout must be mean, concat, max, or sum.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a number.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1).")


class StaticBNT(nn.Module):
    """FC rows are ROI tokens: [B,R,R] -> Transformer -> readout -> MLP.

    Module names and construction order match the supplied GraphTransformer,
    allowing strict weight transfer. The classifier property is an alias, not
    another registered head. return_embedding exposes the input to that head.
    """

    def __init__(self, config: BNTConfig | None = None):
        super().__init__()
        self.config = BNTConfig() if config is None else config
        if not isinstance(self.config, BNTConfig):
            raise TypeError("config must be BNTConfig.")
        cfg = self.config
        self.attention_list = nn.ModuleList()
        self.readout = cfg.readout
        self.node_num = cfg.n_rois
        for _ in range(cfg.n_layers):
            self.attention_list.append(nn.TransformerEncoderLayer(
                d_model=cfg.n_rois, nhead=cfg.nhead, dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout, batch_first=True,
            ))
        final_dim = cfg.n_rois
        if self.readout == "concat":
            self.dim_reduction = nn.Sequential(nn.Linear(cfg.n_rois, 8), nn.LeakyReLU())
            final_dim = 8 * cfg.n_rois
        elif self.readout == "sum":
            self.norm = nn.BatchNorm1d(cfg.n_rois)
        self.embedding_dim = final_dim
        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 32), nn.LeakyReLU(), nn.Linear(32, cfg.num_classes),
        )

    @property
    def classifier(self):
        return self.fc

    def forward(self, fc: torch.Tensor, *, return_embedding=False, return_aux=False):
        if not isinstance(fc, torch.Tensor) or not torch.is_floating_point(fc):
            raise TypeError("FC must be a floating-point tensor.")
        if fc.ndim != 3 or tuple(fc.shape[1:]) != (self.node_num, self.node_num):
            raise ValueError(f"FC must have shape [B,{self.node_num},{self.node_num}].")
        if not torch.isfinite(fc).all():
            raise ValueError("FC contains NaN or Inf.")
        if self.readout == "sum" and self.training and len(fc) < 2:
            raise ValueError("The source's sum readout uses BatchNorm and needs at least two training examples.")
        node_feature = fc
        for attention in self.attention_list:
            node_feature = attention(node_feature)
        if self.readout == "concat":
            embedding = self.dim_reduction(node_feature).reshape((fc.shape[0], -1))
        elif self.readout == "mean":
            embedding = torch.mean(node_feature, dim=1)
        elif self.readout == "max":
            embedding, _ = torch.max(node_feature, dim=1)
        else:
            embedding = self.norm(torch.sum(node_feature, dim=1))
        result = embedding if return_embedding else self.fc(embedding)
        if return_aux:
            return result, {"embedding": embedding, "node_feature": node_feature}
        return result
