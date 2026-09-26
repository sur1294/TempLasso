from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from unified_bn.backbones.base import BaseBackbone


@dataclass(frozen=True)
class VanillaPipelineOutput:
    """
    Standard output of a vanilla static-FC classification pipeline.

    Attributes
    ----------
    logits:
        Unnormalized class scores with shape [B, C].
    embedding:
        Backbone subject embedding with shape [B, D].
    aux:
        Optional backbone diagnostics. Generic training code must not depend
        on model-specific keys in this mapping.
    """

    logits: torch.Tensor
    embedding: torch.Tensor
    aux: Mapping[str, Any] = field(default_factory=dict)


class VanillaPipeline(nn.Module):
    """
    Generic static functional-connectivity classification pipeline.

    Data flow
    ---------
        FC [B,R,R]
            -> BaseBackbone
            -> embedding [B,D]
            -> classifier
            -> logits [B,C]

    Scientific scope
    ----------------
    This class contains no dataset logic, split logic, preprocessing,
    normalization, feature selection, optimization, or metric computation.

    The backbone owns representation learning. The pipeline only composes a
    standardized backbone embedding with a classification head.

    A custom classifier may be supplied when exact reproduction of a
    reference model head is required. Otherwise the default head is a single
    Linear(D, C) layer, providing a deliberately minimal generic baseline.
    """

    def __init__(
        self,
        backbone: BaseBackbone,
        *,
        num_classes: int = 2,
        classifier: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if not isinstance(backbone, BaseBackbone):
            raise TypeError(
                "backbone must implement BaseBackbone, "
                f"got {type(backbone)!r}."
            )

        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")

        self.backbone = backbone
        self.num_classes = int(num_classes)

        if classifier is None:
            classifier = nn.Linear(
                self.backbone.embedding_dim,
                self.num_classes,
            )

        if not isinstance(classifier, nn.Module):
            raise TypeError(
                "classifier must be a torch.nn.Module, "
                f"got {type(classifier)!r}."
            )

        self.classifier = classifier

    @property
    def embedding_dim(self) -> int:
        return self.backbone.embedding_dim

    def forward(
        self,
        fc: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> VanillaPipelineOutput:
        """
        Encode static FC and classify the resulting subject embedding.

        Input validation is delegated to the backbone contract so there is a
        single authoritative validation path for FC tensors.
        """
        backbone_output = self.backbone(fc)
        embedding = backbone_output.embedding
        logits = self.classifier(embedding)

        self._validate_logits(
            logits,
            batch_size=fc.shape[0],
        )

        aux = backbone_output.aux if return_aux else {}

        return VanillaPipelineOutput(
            logits=logits,
            embedding=embedding,
            aux=aux,
        )

    def _validate_logits(
        self,
        logits: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "classifier must return a torch.Tensor, "
                f"got {type(logits)!r}."
            )

        expected = (batch_size, self.num_classes)
        if tuple(logits.shape) != expected:
            raise ValueError(
                "Classifier output contract violated: "
                f"expected {expected}, got {tuple(logits.shape)}."
            )

        if not torch.is_floating_point(logits):
            raise TypeError(
                f"Classifier logits must be floating point, got {logits.dtype}."
            )

        if not torch.isfinite(logits).all():
            raise ValueError("Classifier produced NaN or Inf logits.")
