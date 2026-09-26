#!/usr/bin/env python3
"""
Static ALTER model for the ABIDE-Schaefer100 benchmark.

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
   Yu et al., "ALTER: Adaptive Long-range aware TransformER for brain graphs"
   (brain graph transformer with biased-random-walk long-range encoding).

2. Implementation details:
   Official ALTER release, https://github.com/yushuowiki/ALTER,
   commit ad5c5a95d54aa2a0a2a275c008a692d73af293f7 (2025-09-13),
   files alter/models/LRBGT/lrbgt.py, alter/models/LRBGT/ptdec/*.py,
   alter/models/LRBGT/components/transformer_encoder.py,
   config alter/conf/model/lrbgt.yaml.

   In the released code the model class is called ``BrainNetworkTransformer``
   and is selected with ``model=lrbgt``.

3. Experimental protocol:
   handled outside this file by ``unified_bn``.

WHY THIS FILE EXISTS
--------------------
The anonymous EdgeQuad repository shared by the collaborator
(https://anonymous.4open.science/r/EDGEQUADD-3A6E) advertises seven baseline
model folders in ``zip_all_7_models.sh`` (ALTER, BioBGT, BQN-corrected,
brainetcnn, gcn_gps_experiments, graphtransformer, QUETT) but only uploaded
QUETT and gcn_gps_experiments; the ALTER folder is absent and
``gcn_gps_experiments`` is itself missing the ``models`` package it imports.
ALTER therefore has to come from the public upstream release, which is what
this file vendors. The upstream repository carries no LICENSE file, so the
code is reproduced with explicit attribution and the original docstrings and
comments are preserved.

ARCHITECTURE (defaults = upstream ABIDE configuration, R = 100)
---------------------------------------------------------------
Input: whole-scan FC ``A`` in R^{B x R x R} (signed correlation).

1. RRWP positional encoding (parameter-free, recomputed every forward):

       A+     = A * 1[A > edge_threshold]          # default threshold 0
       P      = D^-1 A+                            # row-stochastic, dead rows -> 0
       pe[:,k]= diag(P^k),  k = 0 .. pos_embed_dim-1   (P^0 = I)

   giving ``[B, R, pos_embed_dim]``, concatenated onto the FC rows:

       X0 = [A ; pe]      -> [B, R, d_model],  d_model = R + pos_embed_dim = 132

2. ``n_layers`` TransPoolingEncoder blocks. Every block is a
   ``TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=1024,
   dropout=0.1, activation=relu, batch_first=True, norm_first=False)``.
   The LAST block additionally applies DEC pooling, mapping R nodes to
   ``cluster_num`` cluster tokens:

       encoder : Linear(d_model*R, 32) -> LeakyReLU -> Linear(32,32)
                 -> LeakyReLU -> Linear(32, d_model*R)
       q       = softmax( (Z C^T)^2 / ||C||_2 )        # frozen orthogonal C
       out     = q^T Z                                 -> [B, cluster_num, d_model]

3. Readout:

       Linear(d_model, 8) + LeakyReLU  -> [B, cluster_num, 8]
       flatten                         -> [B, 8*cluster_num]     <- EMBEDDING
       Linear(8*cluster_num,256) -> LeakyReLU -> Linear(256,32)
                                 -> LeakyReLU -> Linear(32,2)    <- classifier

DEVIATIONS FROM THE RELEASED CODE (all behaviour-preserving; see UPSTREAM.md)
-----------------------------------------------------------------------------
a) The RRWP walk is computed densely and batched instead of building a
   ``torch.sparse_coo_tensor`` per subject inside a Python loop. Verified
   bit-for-bit identical to the released routine (see
   ``tests/test_alter_reference.py``). The dense form is also device-agnostic
   and survives ``torch.use_deterministic_algorithms(True, warn_only=False)``,
   which is this repository's production setting.

b) The released ``add_every_rrwp`` ends with a hard-coded ``.cuda()``. Removed;
   the encoding follows the device and dtype of its input.

c) The released ``sizes: [360, 100]`` in lrbgt.yaml is immediately overwritten
   by ``sizes[0] = node_sz`` in the constructor, so the effective node schedule
   on ABIDE is ``[100, 100]`` and the 360 is dead. This port parameterises the
   same architecture as ``n_layers`` + ``cluster_num`` rather than carrying a
   ``sizes`` list whose first entry is ignored. The defaults reproduce the
   released ABIDE configuration exactly.

d) ``InterpretableTransformerEncoder.__init__`` upstream forwards ``device``
   and ``dtype`` POSITIONALLY to ``nn.TransformerEncoderLayer``. Since torch
   2.0 that signature gained a ``bias`` argument before ``device``, so the
   positional call binds ``device`` to ``bias``. They are passed by keyword
   here. ``_sa_block`` also gained an ``is_causal`` argument, which the
   released override does not accept; it is accepted and ignored here (the
   released code cannot run at all on modern torch without this).

e) The classification head is named ``classifier`` rather than ``fc`` so that
   it matches ``models.bqn.bqn_model.StaticBQN`` and can be consumed by
   ``unified_bn.pipelines.vanilla.VanillaPipeline``. ``fc`` would also collide
   with the "functional connectivity" meaning of ``fc`` everywhere else in
   this repository. State-dict keys change accordingly (``fc.N.*`` ->
   ``classifier.N.*``); nothing else does.

f) ``forward`` takes only FC. The released ``forward(time_seires, node_feature)``
   ignores its first argument -- ALTER never reads time series -- and keeping a
   dead parameter would obscure the data flow. Same decision as StaticBQN.

NOT A DEVIATION, BUT WORTH KNOWING
----------------------------------
* ``need_weights=True`` in ``_sa_block`` forces torch onto the math attention
  path. In train mode this differs from the fused kernel by ~1e-6 in float32.
  The released code has exactly this behaviour and it is preserved. In eval
  mode modern torch takes a fused fast path that bypasses ``_sa_block``
  entirely, so attention weights are only captured during training-mode
  forwards.
* The DEC KL auxiliary loss exists (``dec_kl_loss``) but the released ALTER
  training loop never adds it, and neither does this repository.
* With ``orthogonal_centers=True`` the released Gram-Schmidt leaves cluster
  center row 0 UNNORMALISED while rows 1.. are unit-norm. Preserved verbatim;
  see models/alter/ptdec/cluster.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.alter.ptdec.dec import DEC


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POS_ENCODINGS = ("rrwp", "identity", "none")


@dataclass(frozen=True)
class ALTERConfig:
    """
    Static ALTER architecture.

    Defaults are the released ABIDE configuration
    (``alter/conf/model/lrbgt.yaml`` + ``alter/conf/dataset/ABIDE.yaml``):
    ``pos_encoding='rrwp'``, ``pos_embed_dim=32``, two encoder blocks with the
    second one pooling to 100 clusters, ``nhead=4``, ``dim_feedforward=1024``,
    torch's default transformer ``dropout=0.1``, orthogonal frozen centers and
    projection-style cluster assignment.
    """

    n_rois: int = 100
    n_layers: int = 2
    pos_encoding: str = "rrwp"
    pos_embed_dim: int = 32
    nhead: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    cluster_num: int = 100

    # Which FC entries become walk edges. The released code uses ``data > 0.0``
    # i.e. only positive correlations. Exposed (default 0.0 = released
    # behaviour) because the collaborator's harness comments refer to a 0.3
    # threshold "same as ALTER", which the released code does not implement.
    edge_threshold: float = 0.0

    # Whether the transformer blocks carry bias terms (attention in/out
    # projections, both FFN linears, both LayerNorms). ``True`` is the model
    # the ALTER paper describes and what the released code builds on the
    # torch 1.x it was written for. ``False`` reproduces what the released
    # code actually builds on torch >= 2.0, where its positional
    # ``super().__init__(..., device, dtype)`` call lands ``device=None`` in
    # the ``bias`` slot that torch 2.0 inserted before it -- so every run of
    # the unmodified released code on modern torch is bias-free. The
    # collaborator's harness is such a run.
    transformer_bias: bool = True

    # Released repository defaults.
    orthogonal_centers: bool = True
    freeze_centers: bool = True
    project_assignment: bool = True
    # Output classes of the reference head (2 = the released binary model).
    num_classes: int = 2

    def __post_init__(self) -> None:
        _require_positive_int("num_classes", self.num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        _require_positive_int("n_rois", self.n_rois)
        if self.n_rois <= 1:
            raise ValueError("n_rois must be > 1.")
        _require_positive_int("n_layers", self.n_layers)
        _require_positive_int("nhead", self.nhead)
        _require_positive_int("ffn_dim", self.ffn_dim)
        _require_positive_int("cluster_num", self.cluster_num)

        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(
                f"pos_encoding must be one of {POS_ENCODINGS}; "
                f"got {self.pos_encoding!r}."
            )

        _require_positive_int("pos_embed_dim", self.pos_embed_dim)
        if self.pos_encoding == "rrwp" and self.pos_embed_dim < 2:
            # The released routine unconditionally appends diag(I) and diag(P)
            # before entering its loop, so a walk length below 2 cannot be
            # represented and would silently return 2 columns.
            raise ValueError(
                "pos_encoding='rrwp' requires pos_embed_dim >= 2 "
                "(the walk always contains at least I and D^-1 A)."
            )

        if isinstance(self.dropout, bool) or not isinstance(
            self.dropout, (int, float)
        ):
            raise TypeError("dropout must be numeric.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        if isinstance(self.edge_threshold, bool) or not isinstance(
            self.edge_threshold, (int, float)
        ):
            raise TypeError("edge_threshold must be numeric.")
        if float(self.edge_threshold) < 0.0:
            # Keeps the row-normalisation well defined: with a nonnegative
            # threshold every surviving entry is nonnegative, so the degree is
            # nonnegative and 1/deg is either finite or +inf (which the
            # released code maps to 0).
            raise ValueError("edge_threshold must be nonnegative.")

        for name in (
            "transformer_bias",
            "orthogonal_centers",
            "freeze_centers",
            "project_assignment",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")

        if self.d_model % self.nhead != 0:
            raise ValueError(
                "Transformer d_model must be divisible by nhead: "
                f"d_model={self.d_model} (n_rois={self.n_rois} + "
                f"positional {self.d_model - self.n_rois}), nhead={self.nhead}."
            )

    @property
    def d_model(self) -> int:
        """Transformer width: FC row plus the positional encoding."""
        if self.pos_encoding == "none":
            return int(self.n_rois)
        return int(self.n_rois) + int(self.pos_embed_dim)

    @property
    def embedding_dim(self) -> int:
        return 8 * int(self.cluster_num)


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


# ---------------------------------------------------------------------------
# RRWP long-range positional encoding
# ---------------------------------------------------------------------------

def rrwp_positional_encoding(
    fc: torch.Tensor,
    *,
    walk_length: int,
    edge_threshold: float = 0.0,
) -> torch.Tensor:
    r"""
    Relative random-walk probability encoding of a batch of FC matrices.

    Dense, batched restatement of the released ``add_full_rrwp`` /
    ``add_every_rrwp`` pair. For each subject:

        A+      = A * 1[A > edge_threshold]
        P       = D^-1 A+          (rows with zero degree are left at zero)
        pe[i,k] = (P^k)[i,i],      k = 0 .. walk_length-1,   P^0 = I

    Returns ``[B, R, walk_length]``. Column 0 is identically 1 because
    ``P^0 = I``; every other column is the probability that a degree-weighted
    random walk of that length returns to its start node, which is what makes
    the encoding "long-range aware".

    Parameter-free and recomputed on every forward, exactly as released.
    """
    if not isinstance(fc, torch.Tensor):
        raise TypeError(f"fc must be a torch.Tensor, got {type(fc)!r}.")
    if fc.ndim != 3 or fc.shape[1] != fc.shape[2]:
        raise ValueError(
            f"fc must have shape [B,R,R], got {tuple(fc.shape)}."
        )
    if isinstance(walk_length, bool) or not isinstance(walk_length, int):
        raise TypeError("walk_length must be an integer.")
    if walk_length < 2:
        raise ValueError("walk_length must be >= 2.")

    batch_size, n_nodes, _ = fc.shape
    zero = torch.zeros((), dtype=fc.dtype, device=fc.device)

    adjacency = torch.where(fc > edge_threshold, fc, zero)

    # Released code: deg_inv = 1/deg ; deg_inv[deg_inv == inf] = 0.
    degree_inverse = 1.0 / adjacency.sum(dim=2)
    degree_inverse = torch.where(
        degree_inverse == float("inf"),
        torch.zeros_like(degree_inverse),
        degree_inverse,
    )
    transition = adjacency * degree_inverse.unsqueeze(-1)

    diagonals = [
        # diag(P^0) = diag(I) = 1
        torch.ones(batch_size, n_nodes, dtype=fc.dtype, device=fc.device),
        torch.diagonal(transition, dim1=1, dim2=2),
    ]
    power = transition
    for _ in range(2, walk_length):
        power = torch.bmm(power, transition)
        diagonals.append(torch.diagonal(power, dim1=1, dim2=2))

    return torch.stack(diagonals, dim=-1)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class InterpretableTransformerEncoder(nn.TransformerEncoderLayer):
    """
    ``nn.TransformerEncoderLayer`` that also retains its attention weights.

    Port of ``alter/models/LRBGT/components/transformer_encoder.py``. The only
    functional change relative to a plain encoder layer is that self-attention
    is requested with ``need_weights=True``, which is also what keeps torch on
    the math attention path rather than the fused kernel.

    ``attention_weights`` is populated whenever ``_sa_block`` actually runs.
    On modern torch that is every training-mode forward; in eval mode with no
    mask torch takes a fused fast path that bypasses ``_sa_block``, so the
    attribute keeps whatever the previous training forward stored.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # device/dtype by keyword. Upstream passes them positionally, which on
        # torch >= 2.0 lands ``device=None`` in the ``bias`` slot and silently
        # builds a bias-free layer. That is an explicit ``bias=False`` here, so
        # both variants are reproducible on purpose rather than by accident.
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        # ``is_causal`` is accepted for torch >= 2.0 compatibility. ALTER never
        # uses a causal mask, and the released override predates the argument.
        x, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with optional DEC pooling.

    Input : ``[B, input_node_num, input_feature_size]``
    Output: ``[B, output_node_num, input_feature_size]`` when pooling,
            otherwise the input node count is preserved.

    Port of ``TransPoolingEncoder`` in ``alter/models/LRBGT/lrbgt.py``.
    ``nhead`` and ``dropout`` are hard-coded upstream (4 and torch's 0.1
    default); they are arguments here and their defaults reproduce the
    released model.
    """

    def __init__(
        self,
        input_feature_size: int,
        input_node_num: int,
        hidden_size: int,
        output_node_num: int,
        pooling: bool = True,
        nhead: int = 4,
        dropout: float = 0.1,
        transformer_bias: bool = True,
        orthogonal: bool = True,
        freeze_center: bool = False,
        project_assignment: bool = True,
    ) -> None:
        super().__init__()

        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
            bias=transformer_bias,
        )

        self.pooling = bool(pooling)
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(
                    input_feature_size * input_node_num, encoder_hidden_size
                ),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(
                    encoder_hidden_size, input_feature_size * input_node_num
                ),
            )
            self.dec = DEC(
                cluster_number=output_node_num,
                hidden_dimension=input_feature_size,
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment,
            )

    def is_pooling_enabled(self) -> bool:
        return self.pooling

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.transformer.get_attention_weights()

    def loss(self, assignment: torch.Tensor) -> torch.Tensor:
        """DEC KL auxiliary term. Never used by ALTER's training loop."""
        return self.dec.loss(assignment)


# ---------------------------------------------------------------------------
# Full static ALTER
# ---------------------------------------------------------------------------

class StaticALTER(nn.Module):
    """
    Static ALTER architecture based on the official ALTER release.

    Default input for this project:
        signed whole-scan Schaefer100 FC
        shape [B,100,100]

    This class intentionally accepts ONLY FC in forward(). The released
    repository also passes ROI time series into the model, but the ALTER
    forward computation never uses them.

    Default prediction flow:
        A
        -> X0 = [A ; RRWP(A)]              [B,100,132]
        -> encoder block (no pooling)      [B,100,132]
        -> encoder block + DEC pooling     [B,100,132]  (100 cluster tokens)
        -> Linear(132,8)+LeakyReLU         [B,100,8]
        -> flatten                         [B,800]
        -> 256 -> 32 -> 2
    """

    def __init__(self, config: Optional[ALTERConfig] = None) -> None:
        super().__init__()

        self.config = config if config is not None else ALTERConfig()

        n_rois = self.config.n_rois
        d_model = self.config.d_model

        # Module creation order below mirrors the released constructor so that
        # a shared manual_seed produces the same parameter draw.
        if self.config.pos_encoding == "identity":
            self.node_identity = nn.Parameter(
                torch.zeros(n_rois, self.config.pos_embed_dim),
                requires_grad=True,
            )
            nn.init.kaiming_normal_(self.node_identity)

        # Released schedule for ABIDE is sizes=[R, R] with pooling=[False,True]:
        # node count is preserved until the final block, which pools R nodes
        # into ``cluster_num`` cluster tokens.
        sizes: List[int] = [n_rois] * (self.config.n_layers - 1) + [
            self.config.cluster_num
        ]
        in_sizes: List[int] = [n_rois] + sizes[:-1]
        do_pooling: List[bool] = [False] * (self.config.n_layers - 1) + [True]

        self.attention_list = nn.ModuleList(
            [
                TransPoolingEncoder(
                    input_feature_size=d_model,
                    input_node_num=in_sizes[index],
                    hidden_size=self.config.ffn_dim,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    nhead=self.config.nhead,
                    dropout=self.config.dropout,
                    transformer_bias=self.config.transformer_bias,
                    orthogonal=self.config.orthogonal_centers,
                    freeze_center=self.config.freeze_centers,
                    project_assignment=self.config.project_assignment,
                )
                for index, size in enumerate(sizes)
            ]
        )

        self.dim_reduction = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.LeakyReLU(),
        )

        # Released name is ``fc``; renamed for framework compatibility.
        self.classifier = nn.Sequential(
            nn.Linear(8 * self.config.cluster_num, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, int(self.config.num_classes)),
        )

        self.embedding_dim = self.config.embedding_dim

    # -- forward ------------------------------------------------------------

    def positional_encoding(self, corr: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the positional block appended to each FC row, or None."""
        if self.config.pos_encoding == "none":
            return None
        if self.config.pos_encoding == "identity":
            return self.node_identity.expand(
                corr.shape[0], *self.node_identity.shape
            )
        return rrwp_positional_encoding(
            corr,
            walk_length=self.config.pos_embed_dim,
            edge_threshold=self.config.edge_threshold,
        )

    def forward(
        self,
        corr: torch.Tensor,
        return_embedding: bool = False,
        return_aux: bool = False,
    ):
        self._validate_corr(corr)

        pos_enc = self.positional_encoding(corr)
        node_feature = (
            corr if pos_enc is None else torch.cat([corr, pos_enc], dim=-1)
        )

        assignments: List[Optional[torch.Tensor]] = []
        for block in self.attention_list:
            node_feature, assignment = block(node_feature)
            assignments.append(assignment)

        reduced = self.dim_reduction(node_feature)
        embedding = reduced.reshape(corr.shape[0], -1)

        if return_embedding:
            if return_aux:
                return embedding, self._aux(assignments, pos_enc, node_feature)
            return embedding

        logits = self.classifier(embedding)

        if return_aux:
            aux = self._aux(assignments, pos_enc, node_feature)
            aux["embedding"] = embedding
            return logits, aux

        return logits

    @staticmethod
    def _aux(
        assignments: List[Optional[torch.Tensor]],
        pos_enc: Optional[torch.Tensor],
        node_feature: torch.Tensor,
    ) -> Dict[str, object]:
        pooled = [a for a in assignments if a is not None]
        return {
            "assignment": pooled[-1] if pooled else None,
            "assignments": tuple(assignments),
            "pos_enc": pos_enc,
            "node_feature": node_feature,
        }

    def _validate_corr(self, corr: torch.Tensor) -> None:
        r = self.config.n_rois

        if not isinstance(corr, torch.Tensor):
            raise TypeError(
                f"StaticALTER expects a torch.Tensor, got {type(corr)!r}."
            )

        if corr.ndim != 3:
            raise ValueError(
                f"StaticALTER expects FC shaped [B,R,R], got {tuple(corr.shape)}."
            )

        if corr.shape[1:] != (r, r):
            raise ValueError(
                f"StaticALTER configured for R={r}, got {tuple(corr.shape)}."
            )

        if not torch.is_floating_point(corr):
            raise TypeError(
                f"FC input must be floating point, got dtype={corr.dtype}."
            )

        if not torch.isfinite(corr).all():
            raise ValueError("FC input contains NaN or Inf.")

    # -- diagnostics --------------------------------------------------------

    def get_attention_weights(self) -> List[Optional[Tensor]]:
        return [block.get_attention_weights() for block in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        pooling_blocks = [
            block for block in self.attention_list if block.is_pooling_enabled()
        ]
        if not pooling_blocks:
            raise RuntimeError("This ALTER configuration has no pooling block.")
        return pooling_blocks[-1].dec.get_cluster_centers()

    def dec_kl_loss(self, assignments) -> Optional[torch.Tensor]:
        """
        DEC KL auxiliary objective over the pooling blocks.

        Preserved from the released architecture. ALTER's released training
        loop never adds it to the cross-entropy objective, and neither does
        this repository; it exists so that a future increment which wants the
        auxiliary term does not have to re-derive it.
        """
        pooling_blocks = [
            block for block in self.attention_list if block.is_pooling_enabled()
        ]
        pooled = [a for a in assignments if a is not None]

        total: Optional[torch.Tensor] = None
        for index, assignment in enumerate(pooled):
            term = pooling_blocks[index].loss(assignment)
            total = term if total is None else total + term
        return total

    def parameter_summary(self) -> Dict[str, int]:
        """Registered parameter counts, mirroring StaticBQN.parameter_summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "total": int(total),
            "trainable": int(trainable),
            "attention_blocks": int(
                sum(p.numel() for p in self.attention_list.parameters())
            ),
            "dim_reduction": int(
                sum(p.numel() for p in self.dim_reduction.parameters())
            ),
            "classifier": int(
                sum(p.numel() for p in self.classifier.parameters())
            ),
        }


__all__ = [
    "ALTERConfig",
    "InterpretableTransformerEncoder",
    "POS_ENCODINGS",
    "StaticALTER",
    "TransPoolingEncoder",
    "rrwp_positional_encoding",
]
