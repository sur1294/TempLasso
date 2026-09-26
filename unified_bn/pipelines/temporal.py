from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from unified_bn.backbones.base import BaseBackbone
from unified_bn.temporal.base import BaseTemporalAggregator
from unified_bn.temporal.pooling import lengths_to_mask
from unified_bn.temporal.windowing import (
    WindowPolicyConfig,
    apply_window_policy,
)


@dataclass(frozen=True)
class TemporalPipelineOutput:
    """
    Standard output of the generic temporal classification pipeline.
    """

    logits: torch.Tensor
    temporal_embedding: torch.Tensor
    aux: Mapping[str, Any] = field(default_factory=dict)


class TemporalPipeline(nn.Module):
    """
    Generic dynamic-FC classification pipeline.

    Data flow
    ---------
        dynamic FC [B,K,R,R]
            -> explicit window policy
            -> extract only valid windows [N_valid,R,R]
            -> ONE shared BaseBackbone
            -> scatter embeddings to [B,K,D]
            -> BaseTemporalAggregator
            -> subject embedding [B,D_t]
            -> classifier
            -> logits [B,C]

    Scientific invariants
    ---------------------
    - The same backbone parameters are reused for every valid temporal window.
    - Padded windows are never forwarded through the backbone.
    - The temporal aggregator receives true sequence lengths.
    - Window truncation, when requested, is explicit and deterministic.
    """

    def __init__(
        self,
        backbone: BaseBackbone,
        temporal_aggregator: BaseTemporalAggregator,
        *,
        num_classes: int = 2,
        classifier: Optional[nn.Module] = None,
        window_policy: Optional[WindowPolicyConfig] = None,
    ) -> None:
        super().__init__()

        if not isinstance(backbone, BaseBackbone):
            raise TypeError(
                "backbone must implement BaseBackbone, "
                f"got {type(backbone)!r}."
            )
        if not isinstance(temporal_aggregator, BaseTemporalAggregator):
            raise TypeError(
                "temporal_aggregator must implement BaseTemporalAggregator, "
                f"got {type(temporal_aggregator)!r}."
            )
        if backbone.embedding_dim != temporal_aggregator.input_dim:
            raise ValueError(
                "Backbone embedding dimension and temporal input dimension "
                f"must match: {backbone.embedding_dim} vs "
                f"{temporal_aggregator.input_dim}."
            )
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")

        self.backbone = backbone
        self.temporal_aggregator = temporal_aggregator
        self.num_classes = int(num_classes)
        self.window_policy = (
            window_policy
            if window_policy is not None
            else WindowPolicyConfig()
        )

        if classifier is None:
            classifier = nn.Linear(
                temporal_aggregator.output_dim,
                self.num_classes,
            )
        if not isinstance(classifier, nn.Module):
            raise TypeError("classifier must be a torch.nn.Module.")

        self.classifier = classifier

    def forward(
        self,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> TemporalPipelineOutput:
        self._validate_dynamic_input(dynamic_fc, lengths)

        dynamic_fc, effective_lengths = apply_window_policy(
            dynamic_fc,
            lengths,
            self.window_policy,
        )

        batch_size, max_steps, n_rois, _ = dynamic_fc.shape

        valid_mask = lengths_to_mask(
            effective_lengths,
            max_steps=max_steps,
            device=dynamic_fc.device,
        )

        valid_windows = dynamic_fc[valid_mask]

        if valid_windows.shape[0] == 0:
            raise RuntimeError("No valid temporal windows found.")

        # A single shared backbone processes all real windows in one packed
        # batch. Padding never enters the backbone.
        valid_embeddings = self.backbone.encode(valid_windows)

        window_embeddings = torch.zeros(
            batch_size,
            max_steps,
            self.backbone.embedding_dim,
            dtype=valid_embeddings.dtype,
            device=valid_embeddings.device,
        )
        window_embeddings[valid_mask] = valid_embeddings

        temporal_output = self.temporal_aggregator(
            window_embeddings,
            effective_lengths,
        )

        logits = self.classifier(temporal_output.embedding)
        self._validate_logits(logits, batch_size=batch_size)

        aux: dict[str, Any] = {}

        if return_aux:
            aux = {
                "effective_lengths": effective_lengths,
                "valid_mask": valid_mask,
                "window_embeddings": window_embeddings,
                "n_valid_windows": int(valid_windows.shape[0]),
                "temporal": temporal_output.aux,
            }

        return TemporalPipelineOutput(
            logits=logits,
            temporal_embedding=temporal_output.embedding,
            aux=aux,
        )

    def _validate_dynamic_input(
        self,
        dynamic_fc: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        if not isinstance(dynamic_fc, torch.Tensor):
            raise TypeError("dynamic_fc must be a torch.Tensor.")
        if dynamic_fc.ndim != 4:
            raise ValueError(
                "dynamic_fc must have shape [B,K,R,R], got "
                f"{tuple(dynamic_fc.shape)}."
            )
        if dynamic_fc.shape[2] != dynamic_fc.shape[3]:
            raise ValueError("dynamic_fc must contain square FC matrices.")
        if dynamic_fc.shape[2] != self.backbone.n_rois:
            raise ValueError(
                f"Backbone expects R={self.backbone.n_rois}, got "
                f"R={dynamic_fc.shape[2]}."
            )
        if not torch.is_floating_point(dynamic_fc):
            raise TypeError("dynamic_fc must be floating point.")
        if not torch.isfinite(dynamic_fc).all():
            raise ValueError("dynamic_fc contains NaN or Inf.")

        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a torch.Tensor.")
        if lengths.ndim != 1 or lengths.shape[0] != dynamic_fc.shape[0]:
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
            raise ValueError("All sequence lengths must be positive.")
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
            raise TypeError("classifier must return a torch.Tensor.")

        expected = (batch_size, self.num_classes)
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"Expected classifier output {expected}, got "
                f"{tuple(logits.shape)}."
            )
        if not torch.isfinite(logits).all():
            raise ValueError("Classifier produced NaN or Inf logits.")
