"""
Static BrainNetCNN (Kawahara et al., NeuroImage 2017) for FC-matrix inputs.

Transcribed from the collaborator's harness copy
(``ALTER/models/brainnetcnn/brainnetcnn.py``, itself the BrainNetworkTransformer
repository's baseline), see ``models/brainnetcnn/UPSTREAM.md`` for the
line-by-line provenance and the deviations listed there.

Prediction flow (defaults, Schaefer100):

    A  [B,100,100]
    -> E2E(1 -> 32)   + LeakyReLU(0.33)           [B,32,100,100]
    -> E2E(32 -> 64)  + LeakyReLU(0.33)           [B,64,100,100]
    -> E2N(64 -> 1, (1,R)) + LeakyReLU            [B,1,100,1]
    -> N2G(1 -> 256, (R,1)) + LeakyReLU + Dropout [B,256,1,1]
    -> flatten                                    [B,256]   <- embedding
    -> classifier: 256 -> 128 -> 30 -> 2 (LeakyReLU + Dropout between,
       LeakyReLU after the last layer as in the reference code)

The embedding exposed to the unified pipelines is the 256-d graph-level
vector produced by the N2G layer, i.e. the input of the reference dense
head -- the same "pre-classifier" convention used for StaticBQN and
StaticALTER.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BrainNetCNNConfig:
    n_rois: int = 100
    e2e_channels: Tuple[int, int] = (32, 64)
    n2g_channels: int = 256
    dense_dims: Tuple[int, int] = (128, 30)
    dropout: float = 0.5
    negative_slope: float = 0.33
    # The reference code applies LeakyReLU to the output of the last dense
    # layer (the logits). Kept as the default for fidelity; False gives a
    # plain linear output layer.
    final_activation: bool = True
    # Output classes of the reference head (2 = the harness' binary model).
    num_classes: int = 2

    def __post_init__(self) -> None:
        _require_positive_int("num_classes", self.num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        _require_positive_int("n_rois", self.n_rois)
        if len(self.e2e_channels) != 2:
            raise ValueError("e2e_channels must have exactly two entries.")
        for value in self.e2e_channels:
            _require_positive_int("e2e_channels", value)
        _require_positive_int("n2g_channels", self.n2g_channels)
        if len(self.dense_dims) != 2:
            raise ValueError("dense_dims must have exactly two entries.")
        for value in self.dense_dims:
            _require_positive_int("dense_dims", value)
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a number.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if isinstance(self.negative_slope, bool) or not isinstance(
            self.negative_slope, (int, float)
        ):
            raise TypeError("negative_slope must be a number.")
        if float(self.negative_slope) < 0.0:
            raise ValueError("negative_slope must be nonnegative.")
        if not isinstance(self.final_activation, bool):
            raise TypeError("final_activation must be a bool.")

    @property
    def embedding_dim(self) -> int:
        return int(self.n2g_channels)


class E2EBlock(nn.Module):
    """Edge-to-edge block: a cross-shaped filter summing a row and a column."""

    def __init__(self, in_planes: int, planes: int, roi_num: int, bias: bool = True) -> None:
        super().__init__()
        self.d = int(roi_num)
        self.cnn1 = nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.cnn1(x)  # [B,C,R,1]
        b = self.cnn2(x)  # [B,C,1,R]
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class StaticBrainNetCNN(nn.Module):
    """BrainNetCNN on a batch of FC matrices ``[B,R,R]``."""

    def __init__(self, config: Optional[BrainNetCNNConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else BrainNetCNNConfig()
        c = self.config
        r = c.n_rois
        slope = float(c.negative_slope)
        c1, c2 = c.e2e_channels
        # Module creation order mirrors the reference constructor so a shared
        # manual_seed produces the same parameter draw.
        self.e2econv1 = E2EBlock(1, c1, r, bias=True)
        self.e2econv2 = E2EBlock(c1, c2, r, bias=True)
        self.E2N = nn.Conv2d(c2, 1, (1, r))
        self.N2G = nn.Conv2d(1, c.n2g_channels, (r, 1))
        self.dense1 = nn.Linear(c.n2g_channels, c.dense_dims[0])
        self.dense2 = nn.Linear(c.dense_dims[0], c.dense_dims[1])
        self.dense3 = nn.Linear(c.dense_dims[1], int(c.num_classes))

        self.act = nn.LeakyReLU(negative_slope=slope)
        # nn.Dropout modules (deviation B in UPSTREAM.md): the reference uses
        # F.dropout without ``training=self.training`` and therefore keeps
        # dropping units at evaluation time.
        self.drop_n2g = nn.Dropout(c.dropout)
        head: list[nn.Module] = [
            self.dense1, nn.LeakyReLU(negative_slope=slope), nn.Dropout(c.dropout),
            self.dense2, nn.LeakyReLU(negative_slope=slope), nn.Dropout(c.dropout),
            self.dense3,
        ]
        if c.final_activation:
            head.append(nn.LeakyReLU(negative_slope=slope))
        self.classifier = nn.Sequential(*head)
        self.embedding_dim = c.embedding_dim

    # -- forward ------------------------------------------------------------
    def _validate_fc(self, fc: torch.Tensor) -> None:
        if not isinstance(fc, torch.Tensor):
            raise TypeError(f"fc must be a torch.Tensor, got {type(fc)!r}.")
        if fc.ndim != 3 or fc.shape[1] != fc.shape[2]:
            raise ValueError(f"fc must have shape [B,R,R], got {tuple(fc.shape)}.")
        if fc.shape[1] != self.config.n_rois:
            raise ValueError(
                f"fc has R={fc.shape[1]} but the model was built for R={self.config.n_rois}."
            )
        if not torch.is_floating_point(fc):
            raise TypeError(f"fc must be floating point, got {fc.dtype}.")

    def embed(self, fc: torch.Tensor) -> torch.Tensor:
        self._validate_fc(fc)
        out = fc.unsqueeze(1)                       # [B,1,R,R]
        out = self.act(self.e2econv1(out))
        out = self.act(self.e2econv2(out))
        out = self.act(self.E2N(out))               # [B,1,R,1]
        out = self.drop_n2g(self.act(self.N2G(out)))  # [B,C,1,1]
        return out.reshape(out.shape[0], -1)        # [B,C]

    def forward(
        self,
        fc: torch.Tensor,
        return_embedding: bool = False,
        return_aux: bool = False,
    ):
        embedding = self.embed(fc)
        aux: Dict[str, Any] = {}
        if return_embedding:
            if return_aux:
                return embedding, aux
            return embedding
        logits = self.classifier(embedding)
        if return_aux:
            aux["embedding"] = embedding
            return logits, aux
        return logits


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = ["BrainNetCNNConfig", "E2EBlock", "StaticBrainNetCNN"]
