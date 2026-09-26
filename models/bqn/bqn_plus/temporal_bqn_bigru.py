#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    _BQN_DIR = _Path(__file__).resolve().parents[1]
    if str(_BQN_DIR) not in sys.path:
        sys.path.insert(0, str(_BQN_DIR))

from bqn_model import BQNConfig, StaticBQN
from dynamic_branch import DynamicBranchConfig, DynamicBQNBiGRU


@dataclass(frozen=True)
class TemporalBQNBiGRUConfig:
    # Original BQN-family hyperparameters (architecture itself unchanged)
    bqn_layers: int = 3
    bqn_dropout: float = 0.1
    bqn_activation: str = "leaky_relu"
    cluster_num: int = 4

    # Temporal sequence model
    bigru_hidden_dim: int = 32
    bigru_num_layers: int = 1
    bigru_dropout: float = 0.1
    attention_hidden_dim: int = 32

    # BQN embedding dimension for cluster_num=4
    embedding_dim: int = 32

    # Final classifier
    temporal_layer_norm: bool = True
    classifier_hidden_dim: int = 64
    classifier_dropout: float = 0.1


class TemporalClassifierHead(nn.Module):
    """
    Lightweight subject-level classification head.

    LayerNorm is preferred over BatchNorm here because the dataset is small
    and minibatches contain heterogeneous temporal lengths.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()

        self.norm = (
            nn.LayerNorm(embedding_dim)
            if use_layer_norm
            else nn.Identity()
        )

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor):
        z = self.norm(x)
        return self.head(z), z


class TemporalBQNBiGRUClassifier(nn.Module):
    """
    Dynamic-only model:

        temporal FC windows
            -> original BQN on each VALID window
            -> BiGRU + masked attention
            -> temporal subject embedding
            -> lightweight classifier

    No static whole-scan branch.
    No fusion.
    """

    def __init__(
        self,
        config: Optional[TemporalBQNBiGRUConfig] = None,
    ) -> None:
        super().__init__()

        self.config = (
            config if config is not None else TemporalBQNBiGRUConfig()
        )
        self._validate_config()

        # Original BQN-family backbone. We do not modify its mathematics.
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
                f"BQN embedding_dim={self.bqn.embedding_dim}, "
                f"config.embedding_dim={self.config.embedding_dim}"
            )

        # Dynamic branch uses THIS SAME BQN object on every valid window.
        self.dynamic_branch = DynamicBQNBiGRU(
            shared_bqn=self.bqn,
            config=DynamicBranchConfig(
                hidden_dim=self.config.bigru_hidden_dim,
                num_layers=self.config.bigru_num_layers,
                output_dim=self.config.embedding_dim,
                dropout=self.config.bigru_dropout,
                attention_hidden_dim=self.config.attention_hidden_dim,
                bidirectional=True,
            ),
        )

        self.classifier = TemporalClassifierHead(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.classifier_hidden_dim,
            dropout=self.config.classifier_dropout,
            use_layer_norm=self.config.temporal_layer_norm,
        )

    def _validate_config(self):
        c = self.config

        if c.bqn_layers not in (1, 2, 3, 4, 5):
            raise ValueError("bqn_layers must be in {1,2,3,4,5}")
        if c.bqn_activation not in ("gelu", "leaky_relu", "elu"):
            raise ValueError("Unsupported BQN activation")
        if not 0 <= c.bqn_dropout < 1:
            raise ValueError("Invalid BQN dropout")
        if c.cluster_num <= 0:
            raise ValueError("cluster_num must be positive")

        if c.bigru_hidden_dim <= 0:
            raise ValueError("bigru_hidden_dim must be positive")
        if c.bigru_num_layers <= 0:
            raise ValueError("bigru_num_layers must be positive")
        if not 0 <= c.bigru_dropout < 1:
            raise ValueError("Invalid BiGRU dropout")
        if c.attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be positive")

        if c.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if c.classifier_hidden_dim <= 0:
            raise ValueError("classifier_hidden_dim must be positive")
        if not 0 <= c.classifier_dropout < 1:
            raise ValueError("Invalid classifier dropout")

    def forward(
        self,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        temporal_embedding, dyn_aux = self.dynamic_branch(
            dynamic_fc=dynamic_fc,
            lengths=lengths,
            valid_mask=valid_mask,
            return_aux=True,
        )

        logits, classifier_embedding = self.classifier(
            temporal_embedding
        )

        if logits.shape != (dynamic_fc.shape[0], 2):
            raise RuntimeError(
                f"Unexpected logits shape {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits")

        if return_aux:
            aux: Dict[str, torch.Tensor] = {
                "temporal_embedding": temporal_embedding,
                "classifier_embedding": classifier_embedding,
                "window_embeddings": dyn_aux["window_embeddings"],
                "valid_window_embeddings": dyn_aux[
                    "valid_window_embeddings"
                ],
                "attention_weights": dyn_aux["attention_weights"],
                "sequence_output": dyn_aux["sequence_output"],
                "valid_mask": dyn_aux["valid_mask"],
                "hidden": dyn_aux["hidden"],
                "n_valid_windows": dyn_aux["n_valid_windows"],
            }
            return logits, aux

        return logits

    def parameter_summary(self) -> Dict[str, int]:
        return {
            "total_unique": int(
                sum(p.numel() for p in self.parameters())
            ),
            "trainable_unique": int(
                sum(
                    p.numel()
                    for p in self.parameters()
                    if p.requires_grad
                )
            ),
            "bqn": int(
                sum(p.numel() for p in self.bqn.parameters())
            ),
            "temporal_aggregator": int(
                sum(
                    p.numel()
                    for p in self.dynamic_branch.temporal_aggregator.parameters()
                )
            ),
            "classifier": int(
                sum(p.numel() for p in self.classifier.parameters())
            ),
        }


def _self_test():
    torch.manual_seed(42)

    model = TemporalBQNBiGRUClassifier(
        TemporalBQNBiGRUConfig()
    )
    model.train()

    lengths = torch.tensor([15, 26, 17, 13], dtype=torch.long)
    b = lengths.numel()
    kmax = int(lengths.max())
    valid_mask = (
        torch.arange(kmax).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    x = torch.randn(b, kmax, 100, 100)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)

    d = torch.arange(100)
    x[:, :, d, d] = 1.0

    # Deliberately corrupt padding. It must never enter BQN.
    x = x.clone()
    x[~valid_mask] = 999.0

    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    logits, aux = model(
        dynamic_fc=x,
        lengths=lengths,
        valid_mask=valid_mask,
        return_aux=True,
    )

    assert logits.shape == (b, 2)
    assert aux["temporal_embedding"].shape == (b, 32)
    assert aux["window_embeddings"].shape == (b, kmax, 32)
    assert aux["attention_weights"].shape == (b, kmax)

    assert torch.all(aux["attention_weights"][~valid_mask] == 0)
    assert torch.all(aux["window_embeddings"][~valid_mask] == 0)
    assert torch.allclose(
        aux["attention_weights"].sum(dim=1),
        torch.ones(b),
        atol=1e-6,
    )

    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()

    bqn_grad = any(
        p.grad is not None
        for layer in model.bqn.qp_layers
        for name, p in layer.named_parameters()
        if p.requires_grad and ("mlp_a" in name or "mlp_h" in name)
    )

    bigru_grad = any(
        p.grad is not None
        for p in model.dynamic_branch.temporal_aggregator.gru.parameters()
        if p.requires_grad
    )

    attention_grad = any(
        p.grad is not None
        for p in model.dynamic_branch.temporal_aggregator.attention_pool.parameters()
        if p.requires_grad
    )

    classifier_grad = any(
        p.grad is not None
        for p in model.classifier.parameters()
        if p.requires_grad
    )

    assert model.dynamic_branch.bqn is model.bqn
    assert bqn_grad and bigru_grad and attention_grad and classifier_grad

    print("TemporalBQNBiGRUClassifier self-test: PASS")
    print("literal shared BQN object:", model.dynamic_branch.bqn is model.bqn)
    print("logits:", tuple(logits.shape))
    print("temporal embedding:", tuple(aux["temporal_embedding"].shape))
    print("window embeddings:", tuple(aux["window_embeddings"].shape))
    print("attention weights:", tuple(aux["attention_weights"].shape))
    print(
        "attention sums:",
        aux["attention_weights"].sum(dim=1).detach().tolist(),
    )
    print(
        "padding attention zero:",
        bool(torch.all(aux["attention_weights"][~valid_mask] == 0)),
    )
    print(
        "gradient checks:",
        {
            "bqn": bqn_grad,
            "bigru": bigru_grad,
            "attention": attention_grad,
            "classifier": classifier_grad,
        },
    )
    print("parameter summary:", model.parameter_summary())


if __name__ == "__main__":
    _self_test()
