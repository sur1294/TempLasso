from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from models.bqn.bqn_model import BQNConfig, StaticBQN
from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput


@dataclass(frozen=True)
class BQNBackboneConfig:
    """
    Configuration for adapting the validated StaticBQN implementation to the
    unified backbone interface.

    This adapter does not redefine BQN mathematics. It delegates all model
    computation to `models.bqn.bqn_model.StaticBQN`.

    Parameters mirror the validated reference implementation so the adapter
    can reproduce any already-selected BQN configuration exactly.
    """

    n_rois: int = 100
    n_layers: int = 3
    dropout: float = 0.1
    activation: str = "leaky_relu"
    cluster_num: int = 4

    preserve_unused_mlp_b: bool = True
    orthogonal_centers: bool = True
    freeze_centers: bool = True
    project_assignment: bool = True


class BQNBackbone(BaseBackbone):
    """
    Unified adapter for the validated static BQN reference implementation.

    Contract
    --------
    Input:
        fc : [B, R, R]

    Output:
        BackboneOutput
            embedding : [B, D]

    For the default BQN configuration with four clusters:
        D = 8 * cluster_num = 32

    Scientific invariant
    --------------------
    This class intentionally does not reimplement the BQN forward equations,
    DEC readout, cluster assignment, dimensionality reduction, or classifier.
    It calls the reference `StaticBQN` implementation directly and requests
    its pre-classifier embedding via `return_embedding=True`.

    This preserves the validated BQN architecture while exposing a common
    interface usable by vanilla, temporal, and fusion pipelines.
    """

    def __init__(
        self,
        config: Optional[BQNBackboneConfig] = None,
        *,
        model: Optional[StaticBQN] = None,
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
            config if config is not None else BQNBackboneConfig()
        )

        if model is None:
            reference_config = BQNConfig(
                n_rois=resolved_config.n_rois,
                n_layers=resolved_config.n_layers,
                dropout=resolved_config.dropout,
                activation=resolved_config.activation,
                cluster_num=resolved_config.cluster_num,
                preserve_unused_mlp_b=resolved_config.preserve_unused_mlp_b,
                orthogonal_centers=resolved_config.orthogonal_centers,
                freeze_centers=resolved_config.freeze_centers,
                project_assignment=resolved_config.project_assignment,
                num_classes=int(num_classes),
            )
            resolved_model = StaticBQN(reference_config)
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
        model: StaticBQN,
        config: BQNBackboneConfig,
    ) -> None:
        """
        Ensure an explicitly supplied reference model is structurally
        compatible with the requested adapter configuration.

        This is static because validation must occur before the adapter's
        nn.Module initialization and therefore must not depend on `self`.
        """
        if not isinstance(model, StaticBQN):
            raise TypeError(
                "model must be an instance of models.bqn.bqn_model.StaticBQN, "
                f"got {type(model)!r}."
            )

        expected = {
            "n_rois": config.n_rois,
            "n_layers": config.n_layers,
            "dropout": config.dropout,
            "activation": config.activation,
            "cluster_num": config.cluster_num,
            "preserve_unused_mlp_b": config.preserve_unused_mlp_b,
            "orthogonal_centers": config.orthogonal_centers,
            "freeze_centers": config.freeze_centers,
            "project_assignment": config.project_assignment,
        }

        observed = {
            "n_rois": model.config.n_rois,
            "n_layers": model.config.n_layers,
            "dropout": model.config.dropout,
            "activation": model.config.activation,
            "cluster_num": model.config.cluster_num,
            "preserve_unused_mlp_b": model.config.preserve_unused_mlp_b,
            "orthogonal_centers": model.config.orthogonal_centers,
            "freeze_centers": model.config.freeze_centers,
            "project_assignment": model.config.project_assignment,
        }

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
                "Supplied StaticBQN configuration does not match the adapter "
                f"configuration. Differences: {differences}"
            )

    def _encode_impl(self, fc: torch.Tensor) -> BackboneOutput:
        """
        Delegate encoding to the validated reference BQN implementation.
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
                "topo": aux["topo"],
            },
        )

    def classify(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Return logits from the original reference BQN classifier.

        This method exists for regression/comparability checks. Generic
        unified pipelines should consume `encode(fc)` and attach their own
        pipeline-level classifier.
        """
        return self.reference_model(fc)

    def reference_state_dict(self) -> Mapping[str, Any]:
        """
        Return the underlying validated BQN state dictionary.
        """
        return self.reference_model.state_dict()
