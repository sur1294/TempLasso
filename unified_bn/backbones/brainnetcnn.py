from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Tuple

import torch

from models.brainnetcnn.brainnetcnn_model import BrainNetCNNConfig, StaticBrainNetCNN
from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput


@dataclass(frozen=True)
class BrainNetCNNBackboneConfig:
    """
    Configuration for adapting the transcribed BrainNetCNN to the unified
    backbone interface. Field-for-field mirror of ``BrainNetCNNConfig``
    (see ``models/brainnetcnn/UPSTREAM.md``); defaults are the reference
    architecture.
    """

    n_rois: int = 100
    e2e_channels: Tuple[int, int] = (32, 64)
    n2g_channels: int = 256
    dense_dims: Tuple[int, int] = (128, 30)
    dropout: float = 0.5
    negative_slope: float = 0.33
    final_activation: bool = True

    def __post_init__(self) -> None:
        # Tuples arrive as lists after a JSON round trip (result.json,
        # fingerprint payloads); normalise so equality and hashing hold.
        object.__setattr__(self, "e2e_channels", tuple(int(v) for v in self.e2e_channels))
        object.__setattr__(self, "dense_dims", tuple(int(v) for v in self.dense_dims))
        BrainNetCNNConfig(**asdict(self))

    def to_reference_config(self, num_classes: int = 2) -> BrainNetCNNConfig:
        return BrainNetCNNConfig(**asdict(self), num_classes=int(num_classes))


class BrainNetCNNBackbone(BaseBackbone):
    """
    Unified adapter for ``StaticBrainNetCNN``: ``[B,R,R] -> [B,D]`` with
    ``D = n2g_channels`` (256 by default), the input of the reference dense
    head. Wraps the reference model exactly like ``BQNBackbone`` and
    ``ALTERBackbone`` do, so a vanilla candidate is the reference model and
    temporal/fusion candidates reuse the same encoder per window.
    """

    def __init__(
        self,
        config: Optional[BrainNetCNNBackboneConfig] = None,
        *,
        model: Optional[StaticBrainNetCNN] = None,
        num_classes: int = 2,
    ) -> None:
        resolved_config = config if config is not None else BrainNetCNNBackboneConfig()
        if model is None:
            resolved_model = StaticBrainNetCNN(resolved_config.to_reference_config(num_classes=num_classes))
        else:
            if not isinstance(model, StaticBrainNetCNN):
                raise TypeError(
                    "model must be a models.brainnetcnn.brainnetcnn_model.StaticBrainNetCNN, "
                    f"got {type(model)!r}."
                )
            if model.config != resolved_config.to_reference_config(num_classes=model.config.num_classes):
                raise ValueError(
                    "Supplied StaticBrainNetCNN configuration does not match the "
                    f"adapter configuration: {model.config} vs {resolved_config}."
                )
            resolved_model = model

        super().__init__(
            n_rois=resolved_model.config.n_rois,
            embedding_dim=resolved_model.embedding_dim,
        )
        self.config = resolved_config
        self.reference_model = resolved_model

    def _encode_impl(self, fc: torch.Tensor) -> BackboneOutput:
        embedding = self.reference_model(fc, return_embedding=True)
        return BackboneOutput(embedding=embedding, aux={})

    def classify(self, fc: torch.Tensor) -> torch.Tensor:
        """Logits from the reference dense head (regression checks only)."""
        return self.reference_model(fc)

    def reference_state_dict(self) -> Mapping[str, Any]:
        return self.reference_model.state_dict()

    def extra_repr(self) -> str:
        return (
            f"n_rois={self.n_rois}, embedding_dim={self.embedding_dim}, "
            f"dropout={self.config.dropout}"
        )


__all__ = ["BrainNetCNNBackbone", "BrainNetCNNBackboneConfig"]
