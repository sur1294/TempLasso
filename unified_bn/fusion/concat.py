from __future__ import annotations

import torch

from unified_bn.core.types import FusionOutput
from unified_bn.fusion.base import BaseFusion


class ConcatenationFusion(BaseFusion):
    """
    Parameter-free static/temporal feature concatenation.

    Given
        h_static   : [B, D_s]
        h_temporal : [B, D_t]

    returns
        [h_static || h_temporal] : [B, D_s + D_t]

    Scientific role
    ---------------
    This is the minimal fusion baseline. It introduces no hidden projection,
    gate, attention mechanism, normalization, or additional trainable
    parameters inside the fusion operator itself. Any classification capacity
    after fusion belongs explicitly to the downstream classifier.

    This makes the contribution of combining static and temporal
    representations easy to interpret and provides a controlled reference
    before testing more expressive fusion modules.
    """

    def __init__(self, *, static_dim: int, temporal_dim: int) -> None:
        super().__init__(
            static_dim=static_dim,
            temporal_dim=temporal_dim,
            output_dim=static_dim + temporal_dim,
        )

    def _fuse_impl(
        self,
        h_static: torch.Tensor,
        h_temporal: torch.Tensor,
    ) -> FusionOutput:
        embedding = torch.cat(
            (h_static, h_temporal),
            dim=-1,
        )

        return FusionOutput(
            embedding=embedding,
            aux={},
        )
