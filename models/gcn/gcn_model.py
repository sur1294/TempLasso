"""Dense Kipf-Welling graph convolutions adapted to subject classification.

Each FC matrix is one ROI graph. Signed FC rows are node features; only
positive off-diagonal correlations form message-passing edges. See METHOD.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class GCNConfig:
    n_rois: int = 100
    n_layers: int = 2
    hidden_dim: int = 64
    dropout: float = 0.1
    adjacency: str = "positive"
    node_features: str = "fc_rows"
    readout: str = "mean"
    num_classes: int = 2

    def __post_init__(self):
        for name in ("n_rois", "n_layers", "hidden_dim", "num_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a number.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1).")
        for name, expected in (("adjacency", "positive"), ("node_features", "fc_rows"), ("readout", "mean")):
            if getattr(self, name) != expected:
                raise ValueError(f"This GCN baseline requires {name}={expected!r}.")


def normalized_positive_adjacency(fc: torch.Tensor) -> torch.Tensor:
    """Return D^-1/2 (positive_offdiag(R) + I) D^-1/2 per graph.

    No fitted statistics, cohort graph, threshold search or adjacency cache.
    Correlation signs remain in the separate node-feature tensor, fc.
    """
    if not isinstance(fc, torch.Tensor) or not torch.is_floating_point(fc):
        raise TypeError("FC must be a floating-point tensor.")
    if fc.ndim != 3 or fc.shape[1] != fc.shape[2] or fc.shape[1] < 1:
        raise ValueError("FC must have shape [B,R,R] with positive R.")
    if not torch.isfinite(fc).all():
        raise ValueError("FC contains NaN or Inf.")
    if torch.any(fc.abs() > 1.0 + 1e-5):
        raise ValueError("GCN expects correlations in [-1,1], not Fisher-z values.")
    if not torch.allclose(fc, fc.transpose(-1, -2), atol=1e-5, rtol=0):
        raise ValueError("FC must be symmetric.")
    diagonal = fc.diagonal(dim1=-2, dim2=-1)
    if not torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5, rtol=0):
        raise ValueError("Correlation FC must have a unit diagonal.")

    eye = torch.eye(fc.shape[-1], dtype=fc.dtype, device=fc.device)
    # Remove the existing diagonal before inserting exactly one unit self-loop.
    adjacency = ((fc + fc.transpose(-1, -2)) * 0.5).clamp_min(0)
    adjacency = adjacency * (1 - eye) + eye
    inv_sqrt_degree = adjacency.sum(dim=-1).rsqrt()
    return inv_sqrt_degree.unsqueeze(-1) * adjacency * inv_sqrt_degree.unsqueeze(-2)


class DenseGCNLayer(nn.Module):
    """A_hat H W + b, with bias added AFTER normalized message passing."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor):
        return torch.bmm(adjacency, self.linear(features)) + self.bias


class StaticGCN(nn.Module):
    """FC -> GCN/ReLU/dropout layers -> mean over ROIs -> linear classifier.

    Mean pooling is spatial, within a single scan/window. The temporal pipeline
    calls this encoder on each valid window and retains its sequence order.
    """

    def __init__(self, config: GCNConfig | None = None):
        super().__init__()
        self.config = GCNConfig() if config is None else config
        if not isinstance(self.config, GCNConfig):
            raise TypeError("config must be GCNConfig.")
        cfg = self.config
        self.embedding_dim = cfg.hidden_dim
        self.layers = nn.ModuleList([
            DenseGCNLayer(cfg.n_rois if index == 0 else cfg.hidden_dim, cfg.hidden_dim)
            for index in range(cfg.n_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(self.embedding_dim, cfg.num_classes)

    def forward(self, fc: torch.Tensor, *, return_embedding=False, return_aux=False):
        adjacency = normalized_positive_adjacency(fc)
        if fc.shape[1] != self.config.n_rois:
            raise ValueError(f"Expected {self.config.n_rois} ROIs, got {fc.shape[1]}.")
        features = fc
        for layer in self.layers:
            features = self.dropout(torch.relu(layer(features, adjacency)))
        embedding = features.mean(dim=1)
        result = embedding if return_embedding else self.classifier(embedding)
        if return_aux:
            return result, {"embedding": embedding, "node_feature": features,
                            "normalized_adjacency": adjacency}
        return result
