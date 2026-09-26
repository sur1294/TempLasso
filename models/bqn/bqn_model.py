#!/usr/bin/env python3
"""
Research-grade static BQN model for the Atlas Matters ABIDE-Schaefer100 benchmark.

SCOPE
-----
This file contains MODEL DEFINITIONS ONLY.

It does NOT:
- read HDF5 files,
- create or load train/val/test splits,
- use temporal FC,
- apply Fisher-z,
- normalize FC,
- train the model,
- compute metrics.

REFERENCE HIERARCHY
-------------------
1. Mathematical model:
   Yang et al., "Do We Really Need Message Passing in Brain Network Modeling?",
   ICML 2025, especially Eq. (9).

2. Implementation details:
   Official BQN-demo repository released by the authors.

3. Experimental protocol:
   Atlas Matters benchmark (handled outside this file).

IMPORTANT AUDIT RESULT
----------------------
The official BQN-demo implementation uses:

    MLP_R = Linear(R, R)
    feature_temp = MLP_R(corr)

for corr shaped [B, R, R].

Because torch.nn.Linear acts on the last dimension, this is a row-wise
R -> R transform and corresponds to the paper's A W_A term.

The separately supplied "BQN-corrected" repository changes MLP_R to
Linear(R*R, R), flattening the whole FC matrix and broadcasting the output
over ROI rows. That is NOT Eq. (9), so it is intentionally NOT implemented
here.

The official repository also initializes the iterative state as:

    H0 = A

via `topo = corr.clone()`.

This differs from text elsewhere in the paper that mentions H0 = A W0 in an
ablation description. For static baseline reproduction, this implementation
follows the released BQN code exactly at this point and documents the
paper/code discrepancy rather than silently changing it.

ARCHITECTURE
------------
For input A in R^{B x R x R}:

    H0 = A

For each BQN layer l:

    term_A = H_{l-1} ⊙ Linear_A(A)
    term_H = Linear_H(H_{l-1} ⊙ H_{l-1})
    H_l    = Dropout(Activation(term_A + term_H))

This matches the active forward computation in the official BQN-demo.

The repository additionally declares a third Linear layer MLP_B that is never
used in forward(). It is preserved here behind `preserve_unused_mlp_b=True`
because it is part of the released architecture/state/parameter count, but it
has no mathematical effect on predictions.

READOUT
-------
The official repository uses an orthogonal clustering readout:

    H_L [B,R,R]
      -> flatten full H_L per subject [B,R*R]
      -> encoder R*R -> 32 -> 32 -> R*R
      -> reshape [B,R,R]
      -> fixed orthogonal cluster assignment [B,R,K]
      -> assignment^T @ encoded -> [B,K,R]
      -> Linear(R,8) + LeakyReLU
      -> flatten [B,8K]
      -> MLP: 8K -> 256 -> 32 -> 2

No BatchNorm is used in the official BQN-demo model.

Default Schaefer100 settings:
    R = 100
    clusters = 4
    layers = 3
    activation = LeakyReLU
    dropout = 0.1

These are defaults from the released BQN-demo parser, not claims that they
are necessarily the final Atlas-selected Schaefer100 hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BQNConfig:
    n_rois: int = 100
    n_layers: int = 3
    dropout: float = 0.1
    activation: str = "leaky_relu"
    cluster_num: int = 4

    # Reproduction option:
    # the official BQN-demo declares MLP_B but never uses it.
    preserve_unused_mlp_b: bool = True

    # Official repository defaults.
    orthogonal_centers: bool = True
    freeze_centers: bool = True
    project_assignment: bool = True
    # Output classes of the reference head (2 = the published binary model).
    num_classes: int = 2


def _make_activation(name: str) -> nn.Module:
    table = {
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    if name not in table:
        raise ValueError(
            f"Unsupported activation {name!r}. "
            f"Expected one of {sorted(table)}."
        )
    return table[name]()


# ---------------------------------------------------------------------------
# Core BQN operator
# ---------------------------------------------------------------------------

class QuadraticPerceptron(nn.Module):
    r"""
    One BQN layer following the ACTIVE computation of the official BQN-demo.

    Let A be the original FC matrix and H the current representation.

        H_next =
            H ⊙ (A W_A + b_A)
            + (H ⊙ H) W_H + b_H

    followed by activation and dropout.

    With PyTorch tensors [B,R,R], nn.Linear(R,R) is applied to the final
    dimension independently for each ROI row.

    Therefore:
        self.mlp_a(corr)  has shape [B,R,R]
    and is the code-level realization of the paper's A W_A term.

    `mlp_b` is optionally preserved because the official repository declares
    it, although it is not used in forward().
    """

    def __init__(
        self,
        n_rois: int,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        preserve_unused_mlp_b: bool = True,
    ) -> None:
        super().__init__()

        if n_rois <= 0:
            raise ValueError("n_rois must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.n_rois = int(n_rois)

        # Corresponds to A W_A in Eq. (9).
        self.mlp_a = nn.Linear(self.n_rois, self.n_rois, bias=True)

        # Corresponds to (H ⊙ H) W_H in Eq. (9).
        self.mlp_h = nn.Linear(self.n_rois, self.n_rois, bias=True)

        # Present in the official code but inactive in forward().
        self.mlp_b: Optional[nn.Linear]
        if preserve_unused_mlp_b:
            self.mlp_b = nn.Linear(self.n_rois, self.n_rois, bias=True)
        else:
            self.mlp_b = None

        self.activation = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feature: torch.Tensor,
        corr: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(feature, corr)

        feature_from_original_fc = self.mlp_a(corr)
        term_original = feature * feature_from_original_fc

        term_quadratic = self.mlp_h(feature * feature)

        out = term_original + term_quadratic
        out = self.activation(out)
        out = self.dropout(out)

        return out

    def _validate_inputs(
        self,
        feature: torch.Tensor,
        corr: torch.Tensor,
    ) -> None:
        expected = self.n_rois

        if feature.ndim != 3 or corr.ndim != 3:
            raise ValueError(
                "BQN expects feature and corr shaped [B,R,R]; "
                f"got feature={tuple(feature.shape)}, corr={tuple(corr.shape)}."
            )

        if feature.shape != corr.shape:
            raise ValueError(
                f"feature and corr must have identical shape; "
                f"got {tuple(feature.shape)} vs {tuple(corr.shape)}."
            )

        if feature.shape[1:] != (expected, expected):
            raise ValueError(
                f"Model configured for R={expected}, but received "
                f"{tuple(feature.shape)}."
            )

        if not torch.is_floating_point(feature) or not torch.is_floating_point(corr):
            raise TypeError("BQN inputs must be floating-point tensors.")


# ---------------------------------------------------------------------------
# Orthogonal clustering readout from official BQN-demo
# ---------------------------------------------------------------------------

class ClusterAssignment(nn.Module):
    """
    Soft cluster assignment used by the official BQN-demo readout.

    The default path is the repository's projected assignment:

        score = encoded @ centers^T
        score = score^2
        score = score / ||center||
        assignment = softmax(score)

    Cluster centers are Xavier initialized and, by default, frozen.

    NOTE:
    The released repository's Gram-Schmidt procedure leaves the first center
    unnormalized. This behavior is preserved here for exact implementation
    fidelity. It is documented rather than silently 'fixed'.
    """

    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
        orthogonal: bool = True,
        freeze_center: bool = True,
        project_assignment: bool = True,
    ) -> None:
        super().__init__()

        if cluster_number <= 0:
            raise ValueError("cluster_number must be positive.")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        self.cluster_number = int(cluster_number)
        self.embedding_dimension = int(embedding_dimension)
        self.alpha = float(alpha)
        self.project_assignment = bool(project_assignment)

        if cluster_centers is None:
            initial = torch.zeros(
                self.cluster_number,
                self.embedding_dimension,
                dtype=torch.float32,
            )
            nn.init.xavier_uniform_(initial)
        else:
            initial = cluster_centers.detach().clone().float()
            expected_shape = (
                self.cluster_number,
                self.embedding_dimension,
            )
            if tuple(initial.shape) != expected_shape:
                raise ValueError(
                    f"cluster_centers must have shape {expected_shape}, "
                    f"got {tuple(initial.shape)}."
                )

        if orthogonal:
            initial = self._reference_gram_schmidt(initial)

        self.cluster_centers = nn.Parameter(
            initial,
            requires_grad=(not freeze_center),
        )

    @staticmethod
    def _project(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        denominator = torch.dot(u, u)
        if denominator == 0:
            raise RuntimeError(
                "Zero vector encountered during cluster-center "
                "orthogonalization."
            )
        return (torch.dot(u, v) / denominator) * u

    def _reference_gram_schmidt(
        self,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preserve the released BQN-demo implementation exactly:
        center 0 is copied without normalization; later centers are normalized.
        """
        orthogonal = torch.zeros_like(initial)
        orthogonal[0] = initial[0]

        working = initial.clone()

        for i in range(1, self.cluster_number):
            projection = torch.zeros_like(working[i])

            for j in range(i):
                projection = projection + self._project(
                    working[j],
                    working[i],
                )

            working[i] = working[i] - projection
            norm = torch.norm(working[i], p=2)

            if norm == 0:
                raise RuntimeError(
                    f"Cluster center {i} collapsed to zero during "
                    "orthogonalization."
                )

            orthogonal[i] = working[i] / norm

        return orthogonal

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 2:
            raise ValueError(
                f"ClusterAssignment expects [M,D], got {tuple(batch.shape)}."
            )

        if batch.shape[-1] != self.embedding_dimension:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dimension}, "
                f"got {batch.shape[-1]}."
            )

        if self.project_assignment:
            scores = batch @ self.cluster_centers.T
            scores = torch.pow(scores, 2)

            norms = torch.norm(
                self.cluster_centers,
                p=2,
                dim=-1,
            )

            if torch.any(norms == 0):
                raise RuntimeError("Zero-norm cluster center detected.")

            scores = scores / norms
            return F.softmax(scores, dim=-1)

        # Alternate Student-t assignment present in the official repository.
        norm_squared = torch.sum(
            (batch.unsqueeze(1) - self.cluster_centers) ** 2,
            dim=2,
        )
        numerator = 1.0 / (1.0 + norm_squared / self.alpha)
        numerator = numerator ** ((self.alpha + 1.0) / 2.0)

        denominator = torch.sum(numerator, dim=1, keepdim=True)
        return numerator / denominator


class DECReadout(nn.Module):
    """
    DEC-style orthogonal clustering readout from the official BQN-demo.

    Input:
        topo : [B,R,R]

    Reference flow:
        flatten subject graph        -> [B,R*R]
        encoder                      -> [B,R*R]
        reshape as ROI embeddings    -> [B*R,R]
        cluster assignment           -> [B*R,K]
        reshape                      -> [B,R,K]
        weighted cluster pooling     -> [B,K,R]

    Important:
    The encoder is genuinely used here. This follows the official BQN-demo.
    It does NOT use the lazy per-node projection introduced in the separately
    supplied "BQN-corrected" repository.
    """

    def __init__(
        self,
        n_rois: int,
        cluster_number: int = 4,
        encoder_hidden_size: int = 32,
        alpha: float = 1.0,
        orthogonal: bool = True,
        freeze_center: bool = True,
        project_assignment: bool = True,
    ) -> None:
        super().__init__()

        self.n_rois = int(n_rois)
        self.cluster_number = int(cluster_number)
        self.hidden_dimension = self.n_rois

        flattened_dimension = self.n_rois * self.n_rois

        self.encoder = nn.Sequential(
            nn.Linear(flattened_dimension, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, flattened_dimension),
        )

        self.assignment = ClusterAssignment(
            cluster_number=self.cluster_number,
            embedding_dimension=self.hidden_dimension,
            alpha=alpha,
            orthogonal=orthogonal,
            freeze_center=freeze_center,
            project_assignment=project_assignment,
        )

    def forward(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch.ndim != 3:
            raise ValueError(
                f"DECReadout expects [B,R,R], got {tuple(batch.shape)}."
            )

        batch_size, node_num, feature_dim = batch.shape

        if node_num != self.n_rois or feature_dim != self.n_rois:
            raise ValueError(
                f"DECReadout configured for R={self.n_rois}; "
                f"got {tuple(batch.shape)}."
            )

        flattened = batch.reshape(batch_size, -1)
        encoded = self.encoder(flattened)

        # [B,R*R] -> [B*R,R]
        encoded_nodes = encoded.reshape(
            batch_size * node_num,
            self.hidden_dimension,
        )

        assignment = self.assignment(encoded_nodes)

        # [B*R,K] -> [B,R,K]
        assignment = assignment.reshape(
            batch_size,
            node_num,
            self.cluster_number,
        )

        # [B*R,R] -> [B,R,R]
        encoded_nodes = encoded_nodes.reshape(
            batch_size,
            node_num,
            self.hidden_dimension,
        )

        # [B,K,R]
        node_repr = torch.bmm(
            assignment.transpose(1, 2),
            encoded_nodes,
        )

        return node_repr, assignment


# ---------------------------------------------------------------------------
# Full static BQN
# ---------------------------------------------------------------------------

class StaticBQN(nn.Module):
    """
    Static BQN architecture based on the official BQN-demo implementation.

    Default input for this project:
        signed whole-scan Schaefer100 FC
        shape [B,100,100]

    This class intentionally accepts ONLY FC in forward().
    The official repository also passes ROI time series into the model, but
    the BQN forward computation never uses them. Keeping an unused timeseries
    argument in new research code would obscure the true data flow.

    Default prediction flow:
        A
        -> H0=A
        -> n BQN layers
        -> DEC readout [B,K,R]
        -> Linear(R,8)+LeakyReLU
        -> flatten [B,8K]
        -> 256 -> 32 -> 2
    """

    def __init__(self, config: Optional[BQNConfig] = None) -> None:
        super().__init__()

        self.config = config if config is not None else BQNConfig()

        if self.config.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.config.cluster_num <= 0:
            raise ValueError("cluster_num must be positive.")

        r = self.config.n_rois

        self.qp_layers = nn.ModuleList(
            [
                QuadraticPerceptron(
                    n_rois=r,
                    activation=self.config.activation,
                    dropout=self.config.dropout,
                    preserve_unused_mlp_b=self.config.preserve_unused_mlp_b,
                )
                for _ in range(self.config.n_layers)
            ]
        )

        self.readout = DECReadout(
            n_rois=r,
            cluster_number=self.config.cluster_num,
            encoder_hidden_size=32,
            orthogonal=self.config.orthogonal_centers,
            freeze_center=self.config.freeze_centers,
            project_assignment=self.config.project_assignment,
        )

        # Exact active architecture in the official BQN-demo.
        self.dim_reduction = nn.Sequential(
            nn.Linear(r, 8),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 * self.config.cluster_num, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, int(self.config.num_classes)),
        )

        self.embedding_dim = 8 * self.config.cluster_num

    def forward(
        self,
        corr: torch.Tensor,
        return_embedding: bool = False,
        return_aux: bool = False,
    ):
        self._validate_corr(corr)

        # Official BQN-demo initialization:
        # H0 = A
        topo = corr.clone()

        for layer in self.qp_layers:
            topo = layer(topo, corr)

        graph_level_topo, assignment = self.readout(topo)
        reduced = self.dim_reduction(graph_level_topo)
        embedding = reduced.reshape(corr.shape[0], -1)

        if return_embedding:
            if return_aux:
                return embedding, {
                    "assignment": assignment,
                    "topo": topo,
                }
            return embedding

        logits = self.classifier(embedding)

        if return_aux:
            return logits, {
                "embedding": embedding,
                "assignment": assignment,
                "topo": topo,
            }

        return logits

    def _validate_corr(self, corr: torch.Tensor) -> None:
        r = self.config.n_rois

        if corr.ndim != 3:
            raise ValueError(
                f"StaticBQN expects FC shaped [B,R,R], got {tuple(corr.shape)}."
            )

        if corr.shape[1:] != (r, r):
            raise ValueError(
                f"StaticBQN configured for R={r}, got {tuple(corr.shape)}."
            )

        if not torch.is_floating_point(corr):
            raise TypeError(
                f"FC input must be floating point, got dtype={corr.dtype}."
            )

        if not torch.isfinite(corr).all():
            raise ValueError("FC input contains NaN or Inf.")

    def parameter_summary(self) -> Dict[str, int]:
        """
        Returns registered parameter counts.

        Note that if preserve_unused_mlp_b=True, its parameters are counted
        because they are part of the official released model, even though they
        are inactive in forward().
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total": int(total),
            "trainable": int(trainable),
            "quadratic_layers": int(
                sum(p.numel() for p in self.qp_layers.parameters())
            ),
            "readout": int(
                sum(p.numel() for p in self.readout.parameters())
            ),
            "dim_reduction": int(
                sum(p.numel() for p in self.dim_reduction.parameters())
            ),
            "classifier": int(
                sum(p.numel() for p in self.classifier.parameters())
            ),
        }


# ---------------------------------------------------------------------------
# Local architecture sanity check
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """
    Shape/gradient smoke test only.

    This does not test scientific performance or data preprocessing.
    """
    cfg = BQNConfig(
        n_rois=100,
        n_layers=3,
        dropout=0.1,
        activation="leaky_relu",
        cluster_num=4,
    )

    model = StaticBQN(cfg)
    model.train()

    x = torch.randn(4, 100, 100, dtype=torch.float32)

    logits, aux = model(x, return_aux=True)

    assert logits.shape == (4, 2)
    assert aux["embedding"].shape == (4, 32)
    assert aux["assignment"].shape == (4, 100, 4)
    assert aux["topo"].shape == (4, 100, 100)

    loss = logits.square().mean()
    loss.backward()

    # Verify the two ACTIVE maps in every quadratic layer receive gradients.
    for i, layer in enumerate(model.qp_layers):
        assert layer.mlp_a.weight.grad is not None, f"layer {i} mlp_a has no grad"
        assert layer.mlp_h.weight.grad is not None, f"layer {i} mlp_h has no grad"

        # Official-code artifact: declared but unused.
        if layer.mlp_b is not None:
            assert layer.mlp_b.weight.grad is None, (
                "mlp_b unexpectedly received a gradient; "
                "the forward computation changed."
            )

    print("StaticBQN self-test: PASS")
    print("logits:", tuple(logits.shape))
    print("embedding:", tuple(aux["embedding"].shape))
    print("assignment:", tuple(aux["assignment"].shape))
    print("topo:", tuple(aux["topo"].shape))
    print("parameter summary:", model.parameter_summary())


if __name__ == "__main__":
    _self_test()
