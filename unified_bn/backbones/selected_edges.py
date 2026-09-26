from __future__ import annotations

from dataclasses import dataclass

import torch

from unified_bn.backbones.base import BaseBackbone
from unified_bn.core.types import BackboneOutput
from unified_bn.selectors.lasso import (
    EdgeInputSpace,
    LassoEdgeSelector,
    upper_triangle_edges,
)


@dataclass(frozen=True)
class SelectedEdgeBackboneConfig:
    """
    Candidate-level description of the TempLasso window encoder.

    This is what appears in ``RunIdentity.backbone_config`` with
    ``name='lasso_edges'``. It is a *recipe*, not a materialised edge set:
    the set ``S`` depends on the TRAIN partition and is fit by the runtime for
    each seed, so it is not a hyperparameter and does not belong in the
    candidate fingerprint. The fingerprint of the materialised ``S`` is
    recorded in run provenance instead.
    """

    n_rois: int = 100
    C: float = 0.01
    random_state: int = 0
    max_iter: int = 3000
    input_space: EdgeInputSpace = "fisher_z"
    min_edges: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.n_rois, bool) or not isinstance(self.n_rois, int):
            raise TypeError("n_rois must be an integer.")
        if self.n_rois <= 1:
            raise ValueError("n_rois must be > 1.")


class SelectedEdgeBackbone(BaseBackbone):
    """
    Parameter-free window encoder: ``[B,R,R] -> [B,|S|]``.

    Each window's strict upper triangle is extracted, moved into the
    selector's input space, and reduced to the frozen edge set ``S`` with the
    frozen TRAIN standardisation. There is nothing to learn here by design --
    TempLasso is the minimal realisation of the temporalisation framework,
    in which the window encoder is a fixed, interpretable coordinate
    selection and all sequence modelling capacity lives in the aggregator.

    Because this class satisfies ``BaseBackbone``, it slots into the existing
    ``TemporalPipeline`` unchanged: the pipeline still gathers only valid
    windows, encodes them with this shared module, scatters the ``[N_valid,
    |S|]`` result back to ``[B,K,|S|]``, and hands true lengths to the BiGRU.
    """

    def __init__(self, selector: LassoEdgeSelector, *, n_rois: int) -> None:
        if not isinstance(selector, LassoEdgeSelector):
            raise TypeError(
                "selector must be LassoEdgeSelector, got "
                f"{type(selector)!r}."
            )
        if not selector.is_frozen:
            raise RuntimeError(
                "SelectedEdgeBackbone requires a FROZEN selector. Fit it on the "
                "TRAIN partition and call freeze() first."
            )
        expected_edges = n_rois * (n_rois - 1) // 2
        if selector.n_edges != expected_edges:
            raise ValueError(
                f"Selector was fitted on E={selector.n_edges} edges but "
                f"n_rois={n_rois} implies E={expected_edges}."
            )

        super().__init__(n_rois=n_rois, embedding_dim=selector.n_selected)
        self.input_space: EdgeInputSpace = selector.config.input_space
        self.edge_set_fingerprint = selector.edge_set_fingerprint()

        idx = torch.as_tensor(selector.selected_indices, dtype=torch.long)
        mu = torch.as_tensor(selector.train_mean[idx.numpy()], dtype=torch.float32)
        sd = torch.as_tensor(selector.train_std[idx.numpy()], dtype=torch.float32)
        # Buffers, not Parameters: they follow .to(device) / state_dict but
        # never receive gradients and are never touched by the optimizer.
        self.register_buffer("selected_indices", idx, persistent=True)
        self.register_buffer("train_mean", mu, persistent=True)
        self.register_buffer("train_std", sd, persistent=True)

    def _encode_impl(self, fc: torch.Tensor) -> BackboneOutput:
        edges = upper_triangle_edges(fc, input_space=self.input_space)
        picked = edges.index_select(-1, self.selected_indices)
        embedding = (picked - self.train_mean.to(picked.dtype)) / self.train_std.to(picked.dtype)
        return BackboneOutput(
            embedding=embedding,
            aux={
                "n_selected": int(self.embedding_dim),
                "edge_set_fingerprint": self.edge_set_fingerprint,
            },
        )

    def extra_repr(self) -> str:
        return (
            f"n_rois={self.n_rois}, n_selected={self.embedding_dim}, "
            f"input_space={self.input_space!r}"
        )


__all__ = ["SelectedEdgeBackbone", "SelectedEdgeBackboneConfig"]
