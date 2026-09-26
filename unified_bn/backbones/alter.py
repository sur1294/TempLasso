from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch

from models.alter.alter_model import ALTERConfig, StaticALTER
from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput


@dataclass(frozen=True)
class ALTERBackboneConfig:
    """
    Configuration for adapting the vendored ALTER implementation to the
    unified backbone interface.

    This adapter does not redefine ALTER mathematics. It delegates all model
    computation to `models.alter.alter_model.StaticALTER`, which is a
    bit-for-bit reproduction of the released ALTER model (see
    `models/alter/UPSTREAM.md`).

    Parameters mirror the vendored reference implementation so the adapter
    can reproduce any already-selected ALTER configuration exactly. The
    defaults are the released ABIDE configuration.
    """

    n_rois: int = 100
    n_layers: int = 2
    pos_encoding: str = "rrwp"
    pos_embed_dim: int = 32
    nhead: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    cluster_num: int = 100
    edge_threshold: float = 0.0
    transformer_bias: bool = True

    orthogonal_centers: bool = True
    freeze_centers: bool = True
    project_assignment: bool = True

    def __post_init__(self) -> None:
        # Field-for-field mirror of ALTERConfig, so delegate validation to the
        # reference model's own contract instead of restating it. This makes a
        # malformed candidate fail when the configuration is materialised --
        # in the search space or in the model factory -- rather than deep
        # inside attention at the first forward pass.
        ALTERConfig(**asdict(self))

    def to_reference_config(self, num_classes: int = 2) -> ALTERConfig:
        """Return the equivalent vendored-model configuration."""
        return ALTERConfig(**asdict(self), num_classes=int(num_classes))


class ALTERBackbone(BaseBackbone):
    """
    Unified adapter for the vendored ALTER reference implementation.

    Contract
    --------
    Input:
        fc : [B, R, R]

    Output:
        BackboneOutput
            embedding : [B, D]

    For the released ALTER configuration on Schaefer100:
        D = 8 * cluster_num = 800

    Scientific invariant
    --------------------
    This class intentionally does not reimplement the RRWP long-range
    encoding, the transformer blocks, the DEC cluster pooling, the
    dimensionality reduction, or the classifier. It calls the vendored
    `StaticALTER` implementation directly and requests its pre-classifier
    embedding via `return_embedding=True`.

    This mirrors exactly how `BQNBackbone` wraps `StaticBQN`, so vanilla
    candidates differ between the two baselines only in `backbone_config`.

    Scope
    -----
    Registered for `mode='vanilla'` only. ALTER's forward cost is dominated
    by a `pos_embed_dim`-step dense random walk plus two full self-attention
    layers over R tokens, all recomputed per FC matrix; running it once per
    dynamic window is a separate experiment with its own compute budget, so
    the model factory fails closed rather than silently accepting it in a
    temporal or fusion candidate.
    """

    def __init__(
        self,
        config: Optional[ALTERBackboneConfig] = None,
        *,
        model: Optional[StaticALTER] = None,
        num_classes: int = 2,
    ) -> None:
        # IMPORTANT:
        # Do not assign any nn.Module-valued attribute before BaseBackbone
        # (and therefore torch.nn.Module) has been initialized.
        #
        # Construct/validate the reference model in local variables first,
        # initialize the parent Module, and only then register the model as
        # `self.reference_model`.
        resolved_config = (
            config if config is not None else ALTERBackboneConfig()
        )

        if model is None:
            resolved_model = StaticALTER(
                resolved_config.to_reference_config(num_classes=num_classes)
            )
        else:
            self._validate_reference_model_against_config(
                model=model,
                config=resolved_config,
            )
            resolved_model = model

        super().__init__(
            n_rois=resolved_model.config.n_rois,
            embedding_dim=resolved_model.embedding_dim,
        )

        # Safe now: nn.Module.__init__ has run, so PyTorch can register the
        # wrapped reference model and all of its parameters/submodules.
        self.config = resolved_config
        self.reference_model = resolved_model

    @staticmethod
    def _validate_reference_model_against_config(
        *,
        model: StaticALTER,
        config: ALTERBackboneConfig,
    ) -> None:
        """
        Ensure an explicitly supplied reference model is structurally
        compatible with the requested adapter configuration.

        This is static because validation must occur before the adapter's
        nn.Module initialization and therefore must not depend on `self`.
        """
        if not isinstance(model, StaticALTER):
            raise TypeError(
                "model must be an instance of "
                "models.alter.alter_model.StaticALTER, "
                f"got {type(model)!r}."
            )

        fields = (
            "n_rois",
            "n_layers",
            "pos_encoding",
            "pos_embed_dim",
            "nhead",
            "ffn_dim",
            "dropout",
            "cluster_num",
            "edge_threshold",
            "orthogonal_centers",
            "freeze_centers",
            "project_assignment",
        )

        expected = {name: getattr(config, name) for name in fields}
        observed = {name: getattr(model.config, name) for name in fields}

        if observed != expected:
            differences = {
                key: {
                    "expected": expected[key],
                    "observed": observed[key],
                }
                for key in expected
                if expected[key] != observed[key]
            }
            raise ValueError(
                "Supplied StaticALTER configuration does not match the adapter "
                f"configuration. Differences: {differences}"
            )

    def _encode_impl(self, fc: torch.Tensor) -> BackboneOutput:
        """
        Delegate encoding to the vendored reference ALTER implementation.
        """
        embedding, aux = self.reference_model(
            fc,
            return_embedding=True,
            return_aux=True,
        )

        return BackboneOutput(
            embedding=embedding,
            aux={
                "assignment": aux["assignment"],
                "pos_enc": aux["pos_enc"],
                "node_feature": aux["node_feature"],
            },
        )

    def classify(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Return logits from the original reference ALTER classifier.

        This method exists for regression/comparability checks. Generic
        unified pipelines should consume `encode(fc)` and attach their own
        pipeline-level classifier -- except the vanilla pipeline, which
        deliberately reuses `reference_model.classifier` so that the whole
        pipeline is the reference model.
        """
        return self.reference_model(fc)

    def reference_state_dict(self) -> Mapping[str, Any]:
        """
        Return the underlying vendored ALTER state dictionary.
        """
        return self.reference_model.state_dict()

    def extra_repr(self) -> str:
        return (
            f"n_rois={self.n_rois}, embedding_dim={self.embedding_dim}, "
            f"pos_encoding={self.config.pos_encoding!r}, "
            f"n_layers={self.config.n_layers}"
        )


__all__ = ["ALTERBackbone", "ALTERBackboneConfig"]
