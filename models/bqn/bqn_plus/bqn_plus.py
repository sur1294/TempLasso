#!/usr/bin/env python3
"""
Full BQN+ architecture:
    shared static BQN
    + shared-BQN temporal windows
    + BiGRU temporal aggregation
    + feature-wise gated fusion
    + final classifier

SCOPE
-----
This file contains ONLY the model architecture.

It does NOT:
- read HDF5 files,
- create/load splits,
- train,
- compute metrics,
- perform hyperparameter search.

ARCHITECTURE
------------
Static branch:
    static_fc [B,R,R]
        -> shared BQN
        -> z_static [B,D]

Dynamic branch:
    dynamic_fc [B,K,R,R]
        -> same shared BQN on VALID windows only
        -> window embeddings [B,K,D]
        -> BiGRU + masked attention
        -> z_dynamic [B,D]

Feature-wise gated fusion:
    g = sigmoid(MLP([z_static || z_dynamic]))   [B,D]

    z_fused =
        g * z_static
        + (1-g) * z_dynamic

Classifier:
    z_fused [B,D]
        -> small MLP
        -> logits [B,2]

WEIGHT SHARING
--------------
There is exactly ONE StaticBQN instance:
    self.bqn

The DynamicBQNBiGRU receives that same object by reference.

Therefore:
    self.dynamic_branch.bqn is self.bqn

This is literal parameter sharing, not identical initialization.

IMPORTANT
---------
The original static BQN classifier head is NOT used inside BQN+.
BQN is used as a graph/network encoder via return_embedding=True.

The final BQN+ classifier acts on the fused static/dynamic representation.

AUXILIARY OUTPUTS
-----------------
If return_aux=True, forward returns:
    logits,
    {
        static_embedding,
        dynamic_embedding,
        fused_embedding,
        gate,
        attention_weights,
        window_embeddings,
        sequence_output,
        valid_mask,
    }

These are useful for diagnostics/ablation/interpretability, but the default
training objective should initially remain one final cross-entropy loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

# Support direct execution from models/bqn/bqn_plus/.
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    _BQN_DIR = _Path(__file__).resolve().parents[1]
    if str(_BQN_DIR) not in sys.path:
        sys.path.insert(0, str(_BQN_DIR))

from bqn_model import BQNConfig, StaticBQN
from dynamic_branch import DynamicBranchConfig, DynamicBQNBiGRU


@dataclass(frozen=True)
class BQNPlusConfig:
    # BQN backbone configuration.
    bqn_layers: int = 3
    bqn_dropout: float = 0.1
    bqn_activation: str = "leaky_relu"
    cluster_num: int = 4

    # BiGRU temporal configuration.
    bigru_hidden_dim: int = 32
    bigru_num_layers: int = 1
    bigru_dropout: float = 0.1
    bigru_attention_hidden_dim: int = 32

    # Common embedding dimension.
    embedding_dim: int = 32

    # Fusion/classifier.
    fusion_hidden_dim: int = 64
    classifier_hidden_dim: int = 64
    classifier_dropout: float = 0.1


class FeatureWiseGatedFusion(nn.Module):
    """
    Learned feature-wise fusion of static and dynamic embeddings.

    Given:
        z_static  [B,D]
        z_dynamic [B,D]

    compute:
        g = sigmoid(MLP([z_static || z_dynamic])) [B,D]

        z_fused =
            g * z_static + (1-g) * z_dynamic

    Interpretation:
        g_j near 1 -> feature j relies more on static branch
        g_j near 0 -> feature j relies more on dynamic branch

    Gate values are diagnostics, not automatically causal explanations.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.embedding_dim = int(embedding_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

    def forward(
        self,
        z_static: torch.Tensor,
        z_dynamic: torch.Tensor,
    ):
        if z_static.ndim != 2 or z_dynamic.ndim != 2:
            raise ValueError(
                "z_static and z_dynamic must both be [B,D]."
            )

        if z_static.shape != z_dynamic.shape:
            raise ValueError(
                f"Static/dynamic embedding shape mismatch: "
                f"{tuple(z_static.shape)} vs {tuple(z_dynamic.shape)}"
            )

        if z_static.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {z_static.shape[1]}"
            )

        gate_input = torch.cat(
            [z_static, z_dynamic],
            dim=-1,
        )

        gate = torch.sigmoid(
            self.gate_net(gate_input)
        )

        fused = (
            gate * z_static
            + (1.0 - gate) * z_dynamic
        )

        return fused, gate


class BQNPlusBiGRU(nn.Module):
    """
    Full BQN+ model with BiGRU temporal sequence modeling.

    Forward inputs
    --------------
    static_fc:
        [B,R,R]

    dynamic_fc:
        [B,Kmax,R,R]

    lengths:
        [B]

    valid_mask:
        [B,Kmax] bool

    Returns
    -------
    logits:
        [B,2]

    If return_aux=True:
        (logits, aux)
    """

    def __init__(
        self,
        config: Optional[BQNPlusConfig] = None,
    ) -> None:
        super().__init__()

        self.config = (
            config
            if config is not None
            else BQNPlusConfig()
        )

        self._validate_config()

        # ------------------------------------------------------------
        # ONE shared BQN backbone
        # ------------------------------------------------------------
        self.bqn = StaticBQN(
            BQNConfig(
                n_rois=100,
                n_layers=self.config.bqn_layers,
                dropout=self.config.bqn_dropout,
                activation=self.config.bqn_activation,
                cluster_num=self.config.cluster_num,
            )
        )

        if self.bqn.embedding_dim != self.config.embedding_dim:
            raise ValueError(
                f"BQN embedding_dim={self.bqn.embedding_dim}, but "
                f"BQNPlusConfig.embedding_dim={self.config.embedding_dim}. "
                "Keep these consistent."
            )

        # ------------------------------------------------------------
        # Dynamic branch uses the SAME BQN object by reference
        # ------------------------------------------------------------
        self.dynamic_branch = DynamicBQNBiGRU(
            shared_bqn=self.bqn,
            config=DynamicBranchConfig(
                hidden_dim=self.config.bigru_hidden_dim,
                num_layers=self.config.bigru_num_layers,
                output_dim=self.config.embedding_dim,
                dropout=self.config.bigru_dropout,
                attention_hidden_dim=self.config.bigru_attention_hidden_dim,
                bidirectional=True,
            ),
        )

        # ------------------------------------------------------------
        # Learned feature-wise fusion
        # ------------------------------------------------------------
        self.fusion = FeatureWiseGatedFusion(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.fusion_hidden_dim,
        )

        # ------------------------------------------------------------
        # Final classifier on fused representation
        # ------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.embedding_dim,
                self.config.classifier_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(
                self.config.classifier_hidden_dim,
                2,
            ),
        )

    def _validate_config(self) -> None:
        c = self.config

        if c.bqn_layers <= 0:
            raise ValueError("bqn_layers must be positive.")
        if not 0.0 <= c.bqn_dropout < 1.0:
            raise ValueError("bqn_dropout must satisfy 0 <= p < 1.")
        if c.cluster_num <= 0:
            raise ValueError("cluster_num must be positive.")

        if c.bigru_hidden_dim <= 0:
            raise ValueError("bigru_hidden_dim must be positive.")
        if c.bigru_num_layers <= 0:
            raise ValueError("bigru_num_layers must be positive.")
        if not 0.0 <= c.bigru_dropout < 1.0:
            raise ValueError("bigru_dropout must satisfy 0 <= p < 1.")
        if c.bigru_attention_hidden_dim <= 0:
            raise ValueError(
                "bigru_attention_hidden_dim must be positive."
            )

        if c.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if c.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be positive.")
        if c.classifier_hidden_dim <= 0:
            raise ValueError("classifier_hidden_dim must be positive.")
        if not 0.0 <= c.classifier_dropout < 1.0:
            raise ValueError(
                "classifier_dropout must satisfy 0 <= p < 1."
            )

    def _validate_inputs(
        self,
        static_fc: torch.Tensor,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if static_fc.ndim != 3:
            raise ValueError(
                f"static_fc must be [B,R,R], got "
                f"{tuple(static_fc.shape)}"
            )

        if dynamic_fc.ndim != 4:
            raise ValueError(
                f"dynamic_fc must be [B,K,R,R], got "
                f"{tuple(dynamic_fc.shape)}"
            )

        b_static = static_fc.shape[0]
        b_dynamic = dynamic_fc.shape[0]

        if b_static != b_dynamic:
            raise ValueError(
                f"Batch mismatch: static B={b_static}, "
                f"dynamic B={b_dynamic}"
            )

        if lengths.shape != (b_static,):
            raise ValueError(
                f"lengths must be shape ({b_static},), got "
                f"{tuple(lengths.shape)}"
            )

        if valid_mask.shape != dynamic_fc.shape[:2]:
            raise ValueError(
                f"valid_mask must match [B,K]={dynamic_fc.shape[:2]}, "
                f"got {tuple(valid_mask.shape)}"
            )

        if static_fc.shape[1:] != (100, 100):
            raise ValueError(
                f"static_fc must be Schaefer100 [B,100,100], "
                f"got {tuple(static_fc.shape)}"
            )

        if dynamic_fc.shape[2:] != (100, 100):
            raise ValueError(
                f"dynamic_fc must be Schaefer100 [B,K,100,100], "
                f"got {tuple(dynamic_fc.shape)}"
            )

        if not torch.isfinite(static_fc).all():
            raise ValueError("static_fc contains NaN/Inf.")
        if not torch.isfinite(dynamic_fc).all():
            raise ValueError("dynamic_fc contains NaN/Inf.")

    def forward(
        self,
        static_fc: torch.Tensor,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        self._validate_inputs(
            static_fc=static_fc,
            dynamic_fc=dynamic_fc,
            lengths=lengths,
            valid_mask=valid_mask,
        )

        # ------------------------------------------------------------
        # Static branch:
        # same BQN backbone, embedding before original static classifier
        # ------------------------------------------------------------
        z_static = self.bqn(
            static_fc,
            return_embedding=True,
        )

        # ------------------------------------------------------------
        # Dynamic branch:
        # internally calls THE SAME self.bqn object on valid windows
        # ------------------------------------------------------------
        z_dynamic, dynamic_aux = self.dynamic_branch(
            dynamic_fc=dynamic_fc,
            lengths=lengths,
            valid_mask=valid_mask,
            return_aux=True,
        )

        if z_static.shape != z_dynamic.shape:
            raise RuntimeError(
                f"Static/dynamic embeddings differ: "
                f"{tuple(z_static.shape)} vs {tuple(z_dynamic.shape)}"
            )

        # ------------------------------------------------------------
        # Feature-wise gated fusion
        # ------------------------------------------------------------
        z_fused, gate = self.fusion(
            z_static=z_static,
            z_dynamic=z_dynamic,
        )

        # ------------------------------------------------------------
        # Final classification
        # ------------------------------------------------------------
        logits = self.classifier(z_fused)

        if logits.shape != (static_fc.shape[0], 2):
            raise RuntimeError(
                f"Unexpected logits shape {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "BQN+ logits contain NaN/Inf."
            )

        if return_aux:
            aux: Dict[str, torch.Tensor] = {
                "static_embedding": z_static,
                "dynamic_embedding": z_dynamic,
                "fused_embedding": z_fused,
                "gate": gate,
                "attention_weights": dynamic_aux[
                    "attention_weights"
                ],
                "window_embeddings": dynamic_aux[
                    "window_embeddings"
                ],
                "valid_window_embeddings": dynamic_aux[
                    "valid_window_embeddings"
                ],
                "sequence_output": dynamic_aux[
                    "sequence_output"
                ],
                "valid_mask": dynamic_aux[
                    "valid_mask"
                ],
                "hidden": dynamic_aux[
                    "hidden"
                ],
                "n_valid_windows": dynamic_aux[
                    "n_valid_windows"
                ],
            }
            return logits, aux

        return logits

    def parameter_summary(self):
        """
        Count unique registered model parameters.

        Because self.dynamic_branch.bqn is the same Module object as self.bqn,
        PyTorch's named_parameters() de-duplicates it at model level.
        """
        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total_unique": int(total),
            "trainable_unique": int(trainable),
            "shared_bqn": int(
                sum(p.numel() for p in self.bqn.parameters())
            ),
            "temporal_aggregator": int(
                sum(
                    p.numel()
                    for p in self.dynamic_branch.temporal_aggregator.parameters()
                )
            ),
            "fusion": int(
                sum(p.numel() for p in self.fusion.parameters())
            ),
            "classifier": int(
                sum(p.numel() for p in self.classifier.parameters())
            ),
        }


def _self_test() -> None:
    """
    Full architecture shape / sharing / masking / gradient test.
    """
    torch.manual_seed(42)

    model = BQNPlusBiGRU(
        BQNPlusConfig(
            bqn_layers=3,
            bqn_dropout=0.1,
            bqn_activation="leaky_relu",
            cluster_num=4,
            bigru_hidden_dim=32,
            bigru_num_layers=1,
            bigru_dropout=0.1,
            bigru_attention_hidden_dim=32,
            embedding_dim=32,
            fusion_hidden_dim=64,
            classifier_hidden_dim=64,
            classifier_dropout=0.1,
        )
    )

    model.train()

    b = 4
    lengths = torch.tensor(
        [15, 26, 17, 13],
        dtype=torch.long,
    )
    kmax = int(lengths.max().item())

    valid_mask = (
        torch.arange(kmax).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    # Static correlation-like FC.
    static_fc = torch.randn(b, 100, 100)
    static_fc = 0.5 * (
        static_fc + static_fc.transpose(-1, -2)
    )
    static_fc = torch.tanh(static_fc)
    diag = torch.arange(100)
    static_fc[:, diag, diag] = 1.0

    # Dynamic correlation-like FC.
    dynamic_fc = torch.randn(
        b,
        kmax,
        100,
        100,
    )
    dynamic_fc = 0.5 * (
        dynamic_fc + dynamic_fc.transpose(-1, -2)
    )
    dynamic_fc = torch.tanh(dynamic_fc)
    dynamic_fc[:, :, diag, diag] = 1.0

    # Deliberately contaminate padding. The dynamic branch must gather only
    # valid windows before BQN, so these values must never affect output.
    dynamic_fc = dynamic_fc.clone()
    dynamic_fc[~valid_mask] = 999.0

    logits, aux = model(
        static_fc=static_fc,
        dynamic_fc=dynamic_fc,
        lengths=lengths,
        valid_mask=valid_mask,
        return_aux=True,
    )

    assert logits.shape == (b, 2)
    assert aux["static_embedding"].shape == (b, 32)
    assert aux["dynamic_embedding"].shape == (b, 32)
    assert aux["fused_embedding"].shape == (b, 32)
    assert aux["gate"].shape == (b, 32)
    assert aux["attention_weights"].shape == (b, kmax)

    # Literal shared BQN object.
    assert model.dynamic_branch.bqn is model.bqn

    # Gates must be valid sigmoid outputs.
    assert torch.all(aux["gate"] >= 0)
    assert torch.all(aux["gate"] <= 1)

    # Padded attention must be exactly zero.
    assert torch.all(
        aux["attention_weights"][~valid_mask] == 0
    )

    # Padded BQN window embeddings must be exactly zero.
    assert torch.all(
        aux["window_embeddings"][~valid_mask] == 0
    )

    # End-to-end gradient test.
    labels = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
    )
    loss = nn.functional.cross_entropy(
        logits,
        labels,
    )
    loss.backward()

    # Active BQN gradients.
    bqn_grad = any(
        p.grad is not None
        for layer in model.bqn.qp_layers
        for name, p in layer.named_parameters()
        if (
            p.requires_grad
            and ("mlp_a" in name or "mlp_h" in name)
        )
    )

    bigru_grad = any(
        p.grad is not None
        for p in model.dynamic_branch.temporal_aggregator.gru.parameters()
        if p.requires_grad
    )

    fusion_grad = any(
        p.grad is not None
        for p in model.fusion.parameters()
        if p.requires_grad
    )

    classifier_grad = any(
        p.grad is not None
        for p in model.classifier.parameters()
        if p.requires_grad
    )

    assert bqn_grad
    assert bigru_grad
    assert fusion_grad
    assert classifier_grad

    print("BQNPlusBiGRU self-test: PASS")
    print(
        "literal shared BQN object:",
        model.dynamic_branch.bqn is model.bqn,
    )
    print("logits:", tuple(logits.shape))
    print(
        "static embedding:",
        tuple(aux["static_embedding"].shape),
    )
    print(
        "dynamic embedding:",
        tuple(aux["dynamic_embedding"].shape),
    )
    print(
        "fused embedding:",
        tuple(aux["fused_embedding"].shape),
    )
    print("gate:", tuple(aux["gate"].shape))
    print(
        "gate mean/min/max:",
        float(aux["gate"].mean()),
        float(aux["gate"].min()),
        float(aux["gate"].max()),
    )
    print(
        "attention sums:",
        aux["attention_weights"].sum(dim=1).detach().tolist(),
    )
    print(
        "padded attention exactly zero:",
        bool(
            torch.all(
                aux["attention_weights"][~valid_mask] == 0
            )
        ),
    )
    print(
        "gradient checks:",
        {
            "shared_bqn": bqn_grad,
            "bigru": bigru_grad,
            "fusion": fusion_grad,
            "classifier": classifier_grad,
        },
    )
    print(
        "parameter summary:",
        model.parameter_summary(),
    )


if __name__ == "__main__":
    _self_test()
