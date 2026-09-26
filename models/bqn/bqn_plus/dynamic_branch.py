#!/usr/bin/env python3
"""
Dynamic BQN+ branch:
    temporal FC windows -> SHARED BQN -> BiGRU -> subject dynamic embedding

SCOPE
-----
This file contains ONLY the dynamic branch.

It does NOT:
- load HDF5 files,
- create splits,
- contain the static branch,
- perform static/dynamic fusion,
- contain the final classifier.

EXPECTED INPUT
--------------
dynamic_fc : float tensor [B, Kmax, R, R]
    Batch-padded temporal correlation matrices produced by temporal_dataset.py.

lengths : long tensor [B]
    Native number of windows K_i for each subject.

valid_mask : bool tensor [B, Kmax]
    True for real windows, False for padding.

ARCHITECTURE
------------
Only VALID windows are sent through BQN:

    dynamic_fc [B,K,R,R]
        -> gather valid windows only
        -> [N_valid,R,R]
        -> SAME BQN backbone
        -> [N_valid,D_bqn]
        -> scatter back to padded sequence
        -> [B,K,D_bqn]
        -> BiGRU temporal aggregator
        -> [B,D_dynamic]

This design is intentional:
- padded zero FC matrices never enter BQN;
- the BiGRU receives correct sequence lengths;
- the BQN object is supplied from outside rather than created internally,
  so the eventual BQN+ model can use ONE literal BQN instance for both
  the static whole-scan FC and all dynamic temporal FC windows.

WEIGHT SHARING
--------------
The constructor requires a BQN module:

    shared_bqn = StaticBQN(...)
    dynamic_branch = DynamicBQNBiGRU(shared_bqn, ...)

Later, the full BQN+ model should call that SAME `shared_bqn` for:
    1. static whole-scan FC;
    2. temporal-window FC.

Do not create separate static_bqn/dynamic_bqn copies.

GRADIENTS
---------
By default, BQN is trainable end-to-end. Gradients from the dynamic branch
flow through:
    classifier/fusion (later)
      -> BiGRU
      -> BQN window embeddings
      -> shared BQN parameters.

No BQN freezing is performed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

# When this file is run directly from models/bqn/bqn_plus/, Python normally
# sees only that directory first. Add the parent bqn directory so the existing
# static model can be imported without duplicating code.
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    _BQN_DIR = _Path(__file__).resolve().parents[1]
    if str(_BQN_DIR) not in sys.path:
        sys.path.insert(0, str(_BQN_DIR))

from bqn_model import BQNConfig, StaticBQN
from bigru import BiGRUConfig, BiGRUTemporalAggregator


@dataclass(frozen=True)
class DynamicBranchConfig:
    """
    Configuration for the temporal aggregator only.

    BQN architecture is NOT configured here; it is supplied as an already
    constructed shared backbone.
    """
    hidden_dim: int = 32
    num_layers: int = 1
    output_dim: int = 32
    dropout: float = 0.1
    attention_hidden_dim: int = 32
    bidirectional: bool = True


class DynamicBQNBiGRU(nn.Module):
    """
    Temporal BQN+ dynamic branch.

    Parameters
    ----------
    shared_bqn:
        The SAME StaticBQN object that will later be used by the static branch.
        This module stores a reference to that object; it does not clone it.

    config:
        BiGRU/temporal-aggregation configuration.

    Forward
    -------
    dynamic_fc:
        [B,Kmax,R,R]

    lengths:
        [B]

    valid_mask:
        [B,Kmax]

    return_aux:
        If True, also returns useful research diagnostics:
          - window_embeddings [B,Kmax,D_bqn]
          - valid_window_embeddings [N_valid,D_bqn]
          - attention_weights [B,Kmax]
          - sequence_output [B,Kmax,2H]
          - valid_mask
          - n_valid_windows
    """

    def __init__(
        self,
        shared_bqn: StaticBQN,
        config: Optional[DynamicBranchConfig] = None,
    ) -> None:
        super().__init__()

        if not isinstance(shared_bqn, StaticBQN):
            raise TypeError(
                "shared_bqn must be an existing StaticBQN instance. "
                "Do not pass a copied/independent temporal BQN."
            )

        self.bqn = shared_bqn
        self.config = (
            config if config is not None else DynamicBranchConfig()
        )

        self._validate_config()

        bqn_dim = int(self.bqn.embedding_dim)

        self.temporal_aggregator = BiGRUTemporalAggregator(
            BiGRUConfig(
                input_dim=bqn_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                output_dim=self.config.output_dim,
                dropout=self.config.dropout,
                attention_hidden_dim=self.config.attention_hidden_dim,
                bidirectional=self.config.bidirectional,
            )
        )

        self.bqn_embedding_dim = bqn_dim
        self.output_dim = int(self.config.output_dim)

    def _validate_config(self) -> None:
        c = self.config

        if c.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if c.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if c.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if c.attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be positive.")
        if not 0.0 <= c.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

    def _validate_inputs(
        self,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if dynamic_fc.ndim != 4:
            raise ValueError(
                "dynamic_fc must have shape [B,K,R,R], got "
                f"{tuple(dynamic_fc.shape)}"
            )

        b, kmax, r1, r2 = dynamic_fc.shape
        expected_r = int(self.bqn.config.n_rois)

        if (r1, r2) != (expected_r, expected_r):
            raise ValueError(
                f"Shared BQN expects R={expected_r}, but dynamic_fc has "
                f"shape {tuple(dynamic_fc.shape)}"
            )

        if lengths.ndim != 1 or lengths.shape[0] != b:
            raise ValueError(
                f"lengths must have shape [{b}], got {tuple(lengths.shape)}"
            )

        if valid_mask.shape != (b, kmax):
            raise ValueError(
                f"valid_mask must have shape {(b, kmax)}, got "
                f"{tuple(valid_mask.shape)}"
            )

        if valid_mask.dtype != torch.bool:
            raise TypeError(
                f"valid_mask must be bool, got {valid_mask.dtype}"
            )

        if lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"lengths must be integer tensor, got {lengths.dtype}"
            )

        if not torch.is_floating_point(dynamic_fc):
            raise TypeError(
                f"dynamic_fc must be floating point, got {dynamic_fc.dtype}"
            )

        if not torch.isfinite(dynamic_fc).all():
            raise ValueError("dynamic_fc contains NaN or Inf.")

        if torch.any(lengths <= 0):
            raise ValueError("Every subject must have at least one window.")

        if torch.any(lengths > kmax):
            raise ValueError("A length exceeds Kmax.")

        derived_mask = (
            torch.arange(kmax, device=lengths.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        if not torch.equal(valid_mask, derived_mask):
            raise ValueError(
                "valid_mask does not exactly agree with lengths."
            )

    def encode_valid_windows(
        self,
        dynamic_fc: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """
        Run BQN only on true temporal windows.

        Returns
        -------
        padded_embeddings:
            [B,K,D_bqn], zeros at padded positions.

        valid_embeddings:
            [N_valid,D_bqn]

        flat_valid_indices:
            [N_valid], indices into flattened [B*K] axis.
        """
        b, kmax, r, _ = dynamic_fc.shape

        flat_fc = dynamic_fc.reshape(b * kmax, r, r)
        flat_mask = valid_mask.reshape(b * kmax)

        flat_valid_indices = torch.nonzero(
            flat_mask,
            as_tuple=False,
        ).squeeze(1)

        if flat_valid_indices.numel() == 0:
            raise RuntimeError("Batch contains no valid temporal windows.")

        valid_fc = flat_fc.index_select(
            0,
            flat_valid_indices,
        )

        # Critical operation: use the BQN graph embedding BEFORE its classifier.
        valid_embeddings = self.bqn(
            valid_fc,
            return_embedding=True,
        )

        if valid_embeddings.ndim != 2:
            raise RuntimeError(
                "BQN return_embedding=True must return [N_valid,D]."
            )

        if valid_embeddings.shape[0] != flat_valid_indices.numel():
            raise RuntimeError(
                "Number of BQN embeddings does not match valid windows."
            )

        if valid_embeddings.shape[1] != self.bqn_embedding_dim:
            raise RuntimeError(
                f"Expected BQN embedding dimension {self.bqn_embedding_dim}, "
                f"got {valid_embeddings.shape[1]}"
            )

        # Scatter back into batch/temporal order.
        flat_embeddings = valid_embeddings.new_zeros(
            b * kmax,
            self.bqn_embedding_dim,
        )

        # index_copy is differentiable w.r.t. valid_embeddings, preserving
        # gradient flow back through the shared BQN.
        flat_embeddings = flat_embeddings.index_copy(
            0,
            flat_valid_indices,
            valid_embeddings,
        )

        padded_embeddings = flat_embeddings.reshape(
            b,
            kmax,
            self.bqn_embedding_dim,
        )

        return (
            padded_embeddings,
            valid_embeddings,
            flat_valid_indices,
        )

    def forward(
        self,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        self._validate_inputs(
            dynamic_fc=dynamic_fc,
            lengths=lengths,
            valid_mask=valid_mask,
        )

        (
            window_embeddings,
            valid_window_embeddings,
            flat_valid_indices,
        ) = self.encode_valid_windows(
            dynamic_fc=dynamic_fc,
            valid_mask=valid_mask,
        )

        dynamic_embedding, temporal_aux = self.temporal_aggregator(
            window_embeddings=window_embeddings,
            lengths=lengths,
            valid_mask=valid_mask,
            return_aux=True,
        )

        if dynamic_embedding.shape != (
            dynamic_fc.shape[0],
            self.output_dim,
        ):
            raise RuntimeError(
                "Unexpected dynamic embedding shape: "
                f"{tuple(dynamic_embedding.shape)}"
            )

        if return_aux:
            aux: Dict[str, torch.Tensor] = {
                "window_embeddings": window_embeddings,
                "valid_window_embeddings": valid_window_embeddings,
                "flat_valid_indices": flat_valid_indices,
                "attention_weights": temporal_aux["attention_weights"],
                "sequence_output": temporal_aux["sequence_output"],
                "valid_mask": valid_mask,
                "hidden": temporal_aux["hidden"],
                "n_valid_windows": torch.tensor(
                    flat_valid_indices.numel(),
                    device=dynamic_fc.device,
                    dtype=torch.long,
                ),
            }
            return dynamic_embedding, aux

        return dynamic_embedding

    def parameter_summary(self) -> Dict[str, int]:
        """
        Counts are separated because the BQN parameters are SHARED parameters,
        not parameters unique to the dynamic branch.
        """
        bqn_total = sum(p.numel() for p in self.bqn.parameters())
        temporal_total = sum(
            p.numel()
            for p in self.temporal_aggregator.parameters()
        )

        return {
            "shared_bqn_total": int(bqn_total),
            "temporal_aggregator_total": int(temporal_total),
            "dynamic_path_referenced_total": int(
                bqn_total + temporal_total
            ),
            "temporal_aggregator_trainable": int(
                sum(
                    p.numel()
                    for p in self.temporal_aggregator.parameters()
                    if p.requires_grad
                )
            ),
        }


def _self_test() -> None:
    """
    End-to-end shape/masking/gradient smoke test.

    This creates ONE BQN object and passes it into the dynamic branch,
    demonstrating literal weight sharing.
    """
    torch.manual_seed(42)

    shared_bqn = StaticBQN(
        BQNConfig(
            n_rois=100,
            n_layers=3,
            dropout=0.1,
            activation="leaky_relu",
            cluster_num=4,
        )
    )

    dynamic_branch = DynamicBQNBiGRU(
        shared_bqn=shared_bqn,
        config=DynamicBranchConfig(
            hidden_dim=32,
            num_layers=1,
            output_dim=32,
            dropout=0.1,
            attention_hidden_dim=32,
            bidirectional=True,
        ),
    )

    dynamic_branch.train()

    lengths = torch.tensor(
        [15, 26, 17, 13],
        dtype=torch.long,
    )
    b = lengths.numel()
    kmax = int(lengths.max().item())

    valid_mask = (
        torch.arange(kmax).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    # Correlation-like input.
    x = torch.randn(
        b,
        kmax,
        100,
        100,
    )
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)

    diagonal = torch.arange(100)
    x[:, :, diagonal, diagonal] = 1.0

    # Put deliberately absurd values into padded windows.
    # They must never be sent through BQN because we gather only valid windows.
    x = x.clone()
    x[~valid_mask] = 999.0

    z, aux = dynamic_branch(
        dynamic_fc=x,
        lengths=lengths,
        valid_mask=valid_mask,
        return_aux=True,
    )

    n_valid = int(lengths.sum().item())

    assert z.shape == (b, 32)
    assert aux["window_embeddings"].shape == (
        b,
        kmax,
        shared_bqn.embedding_dim,
    )
    assert aux["valid_window_embeddings"].shape == (
        n_valid,
        shared_bqn.embedding_dim,
    )
    assert aux["attention_weights"].shape == (b, kmax)
    assert int(aux["n_valid_windows"].item()) == n_valid

    # Padded BQN embeddings must remain exactly zero.
    assert torch.all(
        aux["window_embeddings"][~valid_mask] == 0
    )

    # Padded attention must remain exactly zero.
    assert torch.all(
        aux["attention_weights"][~valid_mask] == 0
    )

    # Attention must normalize over valid windows.
    assert torch.allclose(
        aux["attention_weights"].sum(dim=1),
        torch.ones(b),
        atol=1e-6,
    )

    # End-to-end gradient test.
    loss = z.square().mean()
    loss.backward()

    bqn_active_grad = any(
        p.grad is not None
        for layer in shared_bqn.qp_layers
        for name, p in layer.named_parameters()
        if (
            p.requires_grad
            and ("mlp_a" in name or "mlp_h" in name)
        )
    )

    bigru_grad = any(
        p.grad is not None
        for p in dynamic_branch.temporal_aggregator.gru.parameters()
        if p.requires_grad
    )

    attention_grad = any(
        p.grad is not None
        for p in dynamic_branch.temporal_aggregator.attention_pool.parameters()
        if p.requires_grad
    )

    assert bqn_active_grad
    assert bigru_grad
    assert attention_grad

    # Verify it really is the same object, not a copy.
    assert dynamic_branch.bqn is shared_bqn

    print("DynamicBQNBiGRU self-test: PASS")
    print("literal shared BQN object:", dynamic_branch.bqn is shared_bqn)
    print("input dynamic FC:", tuple(x.shape))
    print("lengths:", lengths.tolist())
    print("valid windows:", n_valid)
    print(
        "window embeddings:",
        tuple(aux["window_embeddings"].shape),
    )
    print(
        "valid BQN embeddings:",
        tuple(aux["valid_window_embeddings"].shape),
    )
    print("dynamic embedding:", tuple(z.shape))
    print(
        "attention sums:",
        aux["attention_weights"].sum(dim=1).detach().tolist(),
    )
    print(
        "padded BQN embeddings exactly zero:",
        bool(torch.all(aux["window_embeddings"][~valid_mask] == 0)),
    )
    print(
        "padded attention exactly zero:",
        bool(torch.all(aux["attention_weights"][~valid_mask] == 0)),
    )
    print(
        "gradient checks:",
        {
            "shared_bqn": bqn_active_grad,
            "bigru": bigru_grad,
            "attention": attention_grad,
        },
    )
    print("parameter summary:", dynamic_branch.parameter_summary())


if __name__ == "__main__":
    _self_test()
