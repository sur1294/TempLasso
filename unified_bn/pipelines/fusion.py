from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from unified_bn.backbones.base import BaseBackbone
from unified_bn.fusion.base import BaseFusion
from unified_bn.temporal.base import BaseTemporalAggregator
from unified_bn.temporal.pooling import lengths_to_mask
from unified_bn.temporal.windowing import (
    WindowPolicyConfig,
    apply_window_policy,
)


@dataclass(frozen=True)
class FusionPipelineOutput:
    """
    Standard output of a static + temporal fusion classification pipeline.
    """

    logits: torch.Tensor
    static_embedding: torch.Tensor
    temporal_embedding: torch.Tensor
    fused_embedding: torch.Tensor
    aux: Mapping[str, Any] = field(default_factory=dict)


class FusionPipeline(nn.Module):
    """
    Generic static + temporal FC classification pipeline.

    Data flow
    ---------
        static FC [B,R,R]
            -> static BaseBackbone
            -> h_static [B,D_s]

        dynamic FC [B,K,R,R]
            -> explicit window policy
            -> only valid windows [N_valid,R,R]
            -> temporal BaseBackbone
            -> window embeddings [B,K,D_w]
            -> BaseTemporalAggregator
            -> h_temporal [B,D_t]

        h_static + h_temporal
            -> BaseFusion
            -> h_fused [B,D_f]
            -> classifier
            -> logits [B,C]

    Scientific invariants
    ---------------------
    * Padded temporal windows never enter the temporal backbone.
    * The temporal aggregator receives true effective lengths.
    * Static and temporal subjects must be aligned in one batch.
    * The fusion method is an explicit module rather than hidden inside the
      classifier.
    * Static and temporal backbones are explicit constructor arguments. They
      may later be configured as separate modules or the same shared module;
      this pipeline does not silently impose either scientific choice.
    """

    def __init__(
        self,
        static_backbone: BaseBackbone,
        temporal_backbone: BaseBackbone,
        temporal_aggregator: BaseTemporalAggregator,
        fusion: BaseFusion,
        *,
        num_classes: int = 2,
        classifier: Optional[nn.Module] = None,
        window_policy: Optional[WindowPolicyConfig] = None,
    ) -> None:
        super().__init__()

        if not isinstance(static_backbone, BaseBackbone):
            raise TypeError(
                "static_backbone must implement BaseBackbone, "
                f"got {type(static_backbone)!r}."
            )
        if not isinstance(temporal_backbone, BaseBackbone):
            raise TypeError(
                "temporal_backbone must implement BaseBackbone, "
                f"got {type(temporal_backbone)!r}."
            )
        if not isinstance(
            temporal_aggregator,
            BaseTemporalAggregator,
        ):
            raise TypeError(
                "temporal_aggregator must implement BaseTemporalAggregator, "
                f"got {type(temporal_aggregator)!r}."
            )
        if not isinstance(fusion, BaseFusion):
            raise TypeError(
                "fusion must implement BaseFusion, "
                f"got {type(fusion)!r}."
            )

        if static_backbone.n_rois != temporal_backbone.n_rois:
            raise ValueError(
                "Static and temporal backbones must use the same ROI count: "
                f"{static_backbone.n_rois} vs {temporal_backbone.n_rois}."
            )

        if temporal_backbone.embedding_dim != temporal_aggregator.input_dim:
            raise ValueError(
                "Temporal backbone embedding dimension and temporal "
                "aggregator input dimension must match: "
                f"{temporal_backbone.embedding_dim} vs "
                f"{temporal_aggregator.input_dim}."
            )

        if static_backbone.embedding_dim != fusion.static_dim:
            raise ValueError(
                "Static backbone embedding dimension and fusion static_dim "
                f"must match: {static_backbone.embedding_dim} vs "
                f"{fusion.static_dim}."
            )

        if temporal_aggregator.output_dim != fusion.temporal_dim:
            raise ValueError(
                "Temporal aggregator output dimension and fusion temporal_dim "
                f"must match: {temporal_aggregator.output_dim} vs "
                f"{fusion.temporal_dim}."
            )

        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")

        self.static_backbone = static_backbone
        self.temporal_backbone = temporal_backbone
        self.temporal_aggregator = temporal_aggregator
        self.fusion = fusion
        self.num_classes = int(num_classes)
        self.window_policy = (
            window_policy
            if window_policy is not None
            else WindowPolicyConfig()
        )

        if classifier is None:
            classifier = nn.Linear(
                fusion.output_dim,
                self.num_classes,
            )
        if not isinstance(classifier, nn.Module):
            raise TypeError("classifier must be a torch.nn.Module.")

        self.classifier = classifier

    @property
    def n_rois(self) -> int:
        return self.static_backbone.n_rois

    @property
    def fusion_dim(self) -> int:
        return self.fusion.output_dim

    def forward(
        self,
        static_fc: torch.Tensor,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> FusionPipelineOutput:
        self._validate_joint_input(
            static_fc,
            dynamic_fc,
            lengths,
        )

        static_output = self.static_backbone(static_fc)

        dynamic_fc, effective_lengths = apply_window_policy(
            dynamic_fc,
            lengths,
            self.window_policy,
        )

        batch_size, max_steps, _, _ = dynamic_fc.shape

        valid_mask = lengths_to_mask(
            effective_lengths,
            max_steps=max_steps,
            device=dynamic_fc.device,
        )

        valid_windows = dynamic_fc[valid_mask]

        if valid_windows.shape[0] == 0:
            raise RuntimeError("No valid temporal windows found.")

        valid_embeddings = self.temporal_backbone.encode(
            valid_windows
        )

        window_embeddings = torch.zeros(
            batch_size,
            max_steps,
            self.temporal_backbone.embedding_dim,
            dtype=valid_embeddings.dtype,
            device=valid_embeddings.device,
        )
        window_embeddings[valid_mask] = valid_embeddings

        temporal_output = self.temporal_aggregator(
            window_embeddings,
            effective_lengths,
        )

        fusion_output = self.fusion(
            static_output.embedding,
            temporal_output.embedding,
        )

        logits = self.classifier(
            fusion_output.embedding
        )
        self._validate_logits(
            logits,
            batch_size=batch_size,
        )

        aux: dict[str, Any] = {}

        if return_aux:
            aux = {
                "static": static_output.aux,
                "effective_lengths": effective_lengths,
                "valid_mask": valid_mask,
                "window_embeddings": window_embeddings,
                "n_valid_windows": int(valid_windows.shape[0]),
                "temporal": temporal_output.aux,
                "fusion": fusion_output.aux,
            }

        return FusionPipelineOutput(
            logits=logits,
            static_embedding=static_output.embedding,
            temporal_embedding=temporal_output.embedding,
            fused_embedding=fusion_output.embedding,
            aux=aux,
        )

    def _validate_joint_input(
        self,
        static_fc: torch.Tensor,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        if not isinstance(static_fc, torch.Tensor):
            raise TypeError("static_fc must be a torch.Tensor.")
        if static_fc.ndim != 3:
            raise ValueError(
                "static_fc must have shape [B,R,R], got "
                f"{tuple(static_fc.shape)}."
            )
        if static_fc.shape[1:] != (self.n_rois, self.n_rois):
            raise ValueError(
                f"static_fc expected R={self.n_rois}, got "
                f"{tuple(static_fc.shape)}."
            )
        if not torch.is_floating_point(static_fc):
            raise TypeError("static_fc must be floating point.")
        if not torch.isfinite(static_fc).all():
            raise ValueError("static_fc contains NaN or Inf.")

        if not isinstance(dynamic_fc, torch.Tensor):
            raise TypeError("dynamic_fc must be a torch.Tensor.")
        if dynamic_fc.ndim != 4:
            raise ValueError(
                "dynamic_fc must have shape [B,K,R,R], got "
                f"{tuple(dynamic_fc.shape)}."
            )
        if dynamic_fc.shape[2:] != (self.n_rois, self.n_rois):
            raise ValueError(
                f"dynamic_fc expected R={self.n_rois}, got "
                f"{tuple(dynamic_fc.shape)}."
            )
        if not torch.is_floating_point(dynamic_fc):
            raise TypeError("dynamic_fc must be floating point.")
        if not torch.isfinite(dynamic_fc).all():
            raise ValueError("dynamic_fc contains NaN or Inf.")

        if static_fc.shape[0] != dynamic_fc.shape[0]:
            raise ValueError(
                "Static and temporal inputs must have identical batch size; "
                f"got {static_fc.shape[0]} vs {dynamic_fc.shape[0]}."
            )

        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a torch.Tensor.")
        if (
            lengths.ndim != 1
            or lengths.shape[0] != dynamic_fc.shape[0]
        ):
            raise ValueError("lengths must have shape [B].")
        if lengths.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("lengths must use an integer dtype.")
        if torch.any(lengths < 1):
            raise ValueError(
                "All sequence lengths must be positive."
            )
        if torch.any(lengths > dynamic_fc.shape[1]):
            raise ValueError(
                "A sequence length exceeds the padded temporal dimension."
            )

    def _validate_logits(
        self,
        logits: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "classifier must return a torch.Tensor."
            )

        expected = (batch_size, self.num_classes)
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"Expected classifier output {expected}, got "
                f"{tuple(logits.shape)}."
            )

        if not torch.is_floating_point(logits):
            raise TypeError(
                "Classifier logits must be floating point."
            )

        if not torch.isfinite(logits).all():
            raise ValueError(
                "Classifier produced NaN or Inf logits."
            )
