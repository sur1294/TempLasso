from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import torch.nn as nn

from unified_bn.backbones.alter import ALTERBackbone, ALTERBackboneConfig
from unified_bn.backbones.base import BaseBackbone
from unified_bn.backbones.bqn import BQNBackbone, BQNBackboneConfig
from unified_bn.backbones.bnt import BNTBackbone, BNTBackboneConfig
from unified_bn.backbones.gcn import GCNBackbone, GCNBackboneConfig
from unified_bn.backbones.brainnetcnn import BrainNetCNNBackbone, BrainNetCNNBackboneConfig
from unified_bn.backbones.selected_edges import (
    SelectedEdgeBackbone,
    SelectedEdgeBackboneConfig,
)
from unified_bn.fusion.concat import ConcatenationFusion
from unified_bn.selectors.lasso import LassoEdgeSelector
from unified_bn.pipelines.fusion import FusionPipeline
from unified_bn.pipelines.temporal import TemporalPipeline
from unified_bn.pipelines.vanilla import VanillaPipeline
from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig
from unified_bn.training.config import CandidateRunConfig


class ModelFactoryError(ValueError):
    """Raised when a frozen candidate cannot be mapped to one exact model."""


def build_model_from_candidate(
    run_config: CandidateRunConfig,
) -> nn.Module:
    """
    Construct the Stage-6 pipeline model for one frozen candidate.

    Signature is deliberately ``(run_config)`` only: this function never
    receives data, loaders, or fitted state. TempLasso candidates need a
    TRAIN-fitted edge selector and therefore go through
    :func:`build_templasso_model_from_candidate`; calling this function on
    one fails closed with a message pointing there.
    """
    return _build_model(run_config, edge_selector=None)


def build_templasso_model_from_candidate(
    run_config: CandidateRunConfig,
    *,
    edge_selector: LassoEdgeSelector,
) -> nn.Module:
    """
    Construct a TempLasso model: ``mode='temporal'`` with backbone
    ``name='lasso_edges'`` and a FROZEN, TRAIN-fitted ``edge_selector``.

    Kept separate from :func:`build_model_from_candidate` so the historical
    factory keeps its data-free single-argument contract. This function still
    performs construction only: it validates that the supplied selector is
    frozen and matches the candidate's recipe, and never fits anything.
    """
    if not isinstance(run_config, CandidateRunConfig):
        raise TypeError(
            "run_config must be CandidateRunConfig, got "
            f"{type(run_config)!r}."
        )
    if not _uses_lasso_edges_window_encoder(run_config):
        raise ModelFactoryError(
            "build_templasso_model_from_candidate requires a lasso_edges "
            "window encoder: mode='temporal' with backbone name='lasso_edges', "
            "or mode='fusion' with temporal_backbone_config name='lasso_edges'. "
            f"Got mode={run_config.identity.mode!r}; use "
            "build_model_from_candidate for other candidates."
        )
    if edge_selector is None:
        raise ModelFactoryError(
            "TempLasso requires a FROZEN, TRAIN-fitted edge_selector; got None. "
            "The runtime fits it; the factory never reads data."
        )
    return _build_model(run_config, edge_selector=edge_selector)


def _build_model(
    run_config: CandidateRunConfig,
    *,
    edge_selector: LassoEdgeSelector | None,
) -> nn.Module:
    """
    Shared construction for every Stage-6 family.

    Supported Stage-6K-C model families
    ------------------------------------
    vanilla:
        BQNBackbone -> exact reference StaticBQN classifier

    vanilla with backbone name='alter':
        ALTERBackbone -> exact reference StaticALTER classifier
        Same construction as the BQN vanilla path: the pipeline reuses the
        wrapped reference model's own head, so the resulting module is the
        released ALTER model and nothing else.

    Every static backbone registered in ``_build_backbone`` (``bqn``,
    ``alter``) is accepted wherever a backbone_config appears: as the
    temporal window encoder (TempX), as fusion's static branch, or as both
    fusion branches (X + TempX). The pipelines only require ``BaseBackbone``.

    temporal:
        shared BQNBackbone per valid window
        -> BiGRUAggregator
        -> TemporalPipeline's minimal Linear(D_t, 2) classifier

    temporal with backbone name='lasso_edges'  (TempLasso):
        shared parameter-free SelectedEdgeBackbone per valid window
        -> BiGRUAggregator
        -> TemporalPipeline's minimal Linear(D_t, 2) classifier
        The edge set S is fit on the TRAIN partition by the runtime and
        passed in as a FROZEN ``edge_selector``; this factory never sees
        data. ``edge_selector`` is ignored for every other backbone and
        must be None there.

    fusion:
        independent static BQNBackbone
        + independent temporal BQNBackbone shared across valid windows
        -> BiGRUAggregator
        -> parameter-free ConcatenationFusion
        -> FusionPipeline's minimal Linear(D_s + D_t, 2) classifier

    Scientific boundaries
    ---------------------
    * This function constructs models only.
    * It does not read data or splits.
    * It does not train/evaluate.
    * It does not inspect validation or test metrics.
    * It does not generate hyperparameters.
    * BQN mathematics and BiGRU mathematics are delegated to the already
      validated production implementations.

    Candidate identity is fail-closed: unknown model names or configuration
    keys are rejected instead of being silently ignored.
    """

    if not isinstance(run_config, CandidateRunConfig):
        raise TypeError(
            "run_config must be CandidateRunConfig, got "
            f"{type(run_config)!r}."
        )

    K = int(run_config.identity.num_classes)

    def _bb(payload, **kwargs):
        return _build_backbone(payload, num_classes=K, **kwargs)

    if run_config.identity.mode == "vanilla":
        if run_config.identity.temporal_config is not None:
            raise ModelFactoryError(
                "Vanilla candidate unexpectedly defines temporal_config."
            )
        if run_config.identity.window_policy is not None:
            raise ModelFactoryError(
                "Vanilla candidate unexpectedly defines window_policy."
            )
        if run_config.identity.fusion_config is not None:
            raise ModelFactoryError(
                "Vanilla candidate unexpectedly defines fusion_config."
            )

        _require_no_edge_selector(edge_selector, mode="vanilla")
        if _backbone_name(run_config.identity.backbone_config) == "lasso_edges":
            raise ModelFactoryError(
                "backbone name='lasso_edges' is a temporal window encoder and "
                "has no validated static classifier path; use mode='temporal'."
            )
        backbone = _bb(
            run_config.identity.backbone_config
        )

        # Exact static-BQN reproduction path:
        # BQNBackbone.forward() returns the reference pre-classifier embedding,
        # and the pipeline applies the SAME classifier module owned by the
        # wrapped StaticBQN. No independently initialized replacement head.
        return VanillaPipeline(
            backbone,
            num_classes=K,
            classifier=backbone.reference_model.classifier,
        )

    if run_config.identity.mode == "temporal":
        temporal_mapping = run_config.identity.temporal_config
        if temporal_mapping is None:
            raise ModelFactoryError(
                "Temporal candidate requires temporal_config."
            )
        if run_config.identity.window_policy is None:
            raise ModelFactoryError(
                "Temporal candidate requires window_policy."
            )
        if run_config.identity.fusion_config is not None:
            raise ModelFactoryError(
                "Temporal candidate unexpectedly defines fusion_config."
            )

        backbone = _bb(
            run_config.identity.backbone_config,
            edge_selector=edge_selector,
        )
        if not isinstance(backbone, SelectedEdgeBackbone):
            _require_no_edge_selector(edge_selector, mode="temporal/bqn")
        temporal_aggregator = _build_temporal_aggregator(
            temporal_mapping,
            input_dim=backbone.embedding_dim,
        )

        return TemporalPipeline(
            backbone,
            temporal_aggregator,
            num_classes=K,
            window_policy=run_config.identity.window_policy,
        )

    if run_config.identity.mode == "fusion":
        temporal_mapping = run_config.identity.temporal_config
        fusion_mapping = run_config.identity.fusion_config

        if temporal_mapping is None:
            raise ModelFactoryError(
                "Fusion candidate requires temporal_config."
            )
        if run_config.identity.window_policy is None:
            raise ModelFactoryError(
                "Fusion candidate requires window_policy."
            )
        if fusion_mapping is None:
            raise ModelFactoryError(
                "Fusion candidate requires fusion_config."
            )

        if _backbone_name(run_config.identity.backbone_config) == "lasso_edges":
            raise ModelFactoryError(
                "Fusion's static branch must be a real FC encoder (name='bqn'); "
                "lasso_edges is a temporal window encoder. For X + TempLasso put "
                "it in temporal_backbone_config."
            )

        temporal_payload = run_config.identity.temporal_backbone_config
        if temporal_payload is None:
            # Historical controlled fusion baseline (X + TempX):
            # both branches use the same frozen BQN architecture configuration,
            # but are distinct module instances with independent trainable
            # weights. Sharing weights across static whole-run FC and
            # dynamic-window FC would impose an additional scientific
            # constraint and is therefore not introduced silently.
            _require_no_edge_selector(edge_selector, mode="fusion")
            static_backbone = _bb(
                run_config.identity.backbone_config
            )
            temporal_backbone = _bb(
                run_config.identity.backbone_config
            )
        else:
            # Heterogeneous fusion (X + TempLasso): static branch from
            # backbone_config, temporal window encoder from
            # temporal_backbone_config. Gradients reach every trainable
            # parameter on both branches; the lasso_edges encoder itself has
            # none and acts as a fixed, train-fitted projection.
            static_backbone = _bb(
                run_config.identity.backbone_config
            )
            temporal_backbone = _bb(
                temporal_payload,
                edge_selector=edge_selector,
            )
            if not isinstance(temporal_backbone, SelectedEdgeBackbone):
                _require_no_edge_selector(edge_selector, mode="fusion/bqn-temporal")

        temporal_aggregator = _build_temporal_aggregator(
            temporal_mapping,
            input_dim=temporal_backbone.embedding_dim,
        )
        fusion = _build_fusion(
            fusion_mapping,
            static_dim=static_backbone.embedding_dim,
            temporal_dim=temporal_aggregator.output_dim,
        )

        return FusionPipeline(
            static_backbone,
            temporal_backbone,
            temporal_aggregator,
            fusion,
            num_classes=K,
            window_policy=run_config.identity.window_policy,
        )

    raise ModelFactoryError(
        "Stage 6K-C supports only mode='vanilla', mode='temporal', "
        f"or mode='fusion'; got {run_config.identity.mode!r}."
    )


def _uses_lasso_edges_window_encoder(run_config: CandidateRunConfig) -> bool:
    ident = run_config.identity
    if ident.mode == "temporal":
        return _backbone_name(ident.backbone_config) == "lasso_edges"
    if ident.mode == "fusion" and ident.temporal_backbone_config is not None:
        return _backbone_name(ident.temporal_backbone_config) == "lasso_edges"
    return False


def _backbone_name(payload: Mapping[str, Any]) -> Any:
    if not isinstance(payload, Mapping):
        raise TypeError("backbone_config must be a mapping.")
    return payload.get("name", None)


def _require_no_edge_selector(
    edge_selector: LassoEdgeSelector | None,
    *,
    mode: str,
) -> None:
    if edge_selector is not None:
        raise ModelFactoryError(
            f"edge_selector was supplied for {mode}, but only "
            "temporal candidates with backbone name='lasso_edges' consume one."
        )


def _build_backbone(
    payload: Mapping[str, Any],
    *,
    edge_selector: LassoEdgeSelector | None = None,
    num_classes: int = 2,
) -> BaseBackbone:
    if not isinstance(payload, Mapping):
        raise TypeError("backbone_config must be a mapping.")

    mapping = dict(payload)
    name = mapping.pop("name", None)

    if name == "lasso_edges":
        return _build_selected_edge_backbone(mapping, edge_selector=edge_selector)

    if name == "alter":
        return _build_alter_backbone(mapping, num_classes=num_classes)

    if name == "brainnetcnn":
        return _build_brainnetcnn_backbone(mapping, num_classes=num_classes)

    if name == "gcn":
        _require_exact_keys(mapping, dataclass_type=GCNBackboneConfig, context="GCN backbone")
        try:
            return GCNBackbone(GCNBackboneConfig(**mapping), num_classes=num_classes)
        except (TypeError, ValueError) as exc:
            raise ModelFactoryError(f"Invalid GCN backbone configuration: {exc}") from exc

    if name == "bnt":
        _require_exact_keys(mapping, dataclass_type=BNTBackboneConfig, context="BNT backbone")
        try:
            return BNTBackbone(BNTBackboneConfig(**mapping), num_classes=num_classes)
        except (TypeError, ValueError) as exc:
            raise ModelFactoryError(f"Invalid BNT backbone configuration: {exc}") from exc

    if name != "bqn":
        raise ModelFactoryError(
            "Stage 6K-C currently supports only backbone name='bqn', "
            f"name='alter', name='brainnetcnn', name='bnt', name='gcn' or name='lasso_edges'; got {name!r}."
        )

    _require_exact_keys(
        mapping,
        dataclass_type=BQNBackboneConfig,
        context="BQN backbone",
    )

    try:
        config = BQNBackboneConfig(**mapping)
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(
            "Invalid BQN backbone configuration."
        ) from exc

    return BQNBackbone(config, num_classes=num_classes)


def _build_alter_backbone(mapping: Mapping[str, Any], *, num_classes: int = 2) -> ALTERBackbone:
    _require_exact_keys(
        mapping,
        dataclass_type=ALTERBackboneConfig,
        context="ALTER backbone",
    )

    try:
        config = ALTERBackboneConfig(**mapping)
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(
            "Invalid ALTER backbone configuration."
        ) from exc

    try:
        return ALTERBackbone(config, num_classes=num_classes)
    except (TypeError, ValueError) as exc:
        # ALTERConfig validates architectural consistency (for example that
        # d_model is divisible by nhead), which only surfaces at construction.
        raise ModelFactoryError(
            "Invalid ALTER backbone configuration."
        ) from exc


def _build_brainnetcnn_backbone(mapping: Mapping[str, Any], *, num_classes: int = 2) -> BrainNetCNNBackbone:
    _require_exact_keys(
        mapping,
        dataclass_type=BrainNetCNNBackboneConfig,
        context="BrainNetCNN backbone",
    )
    try:
        config = BrainNetCNNBackboneConfig(**mapping)
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(
            "Invalid BrainNetCNN backbone configuration."
        ) from exc
    return BrainNetCNNBackbone(config, num_classes=num_classes)


def _build_selected_edge_backbone(
    mapping: Mapping[str, Any],
    *,
    edge_selector: LassoEdgeSelector | None,
) -> SelectedEdgeBackbone:
    _require_exact_keys(
        mapping,
        dataclass_type=SelectedEdgeBackboneConfig,
        context="lasso_edges backbone",
    )
    try:
        config = SelectedEdgeBackboneConfig(**mapping)
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(
            "Invalid lasso_edges backbone configuration."
        ) from exc

    if edge_selector is None:
        raise ModelFactoryError(
            "backbone name='lasso_edges' requires a FROZEN LassoEdgeSelector "
            "fitted on the TRAIN partition. The model factory does not read "
            "data; the runtime must fit it and pass edge_selector=."
        )
    if not isinstance(edge_selector, LassoEdgeSelector):
        raise TypeError(
            "edge_selector must be LassoEdgeSelector, got "
            f"{type(edge_selector)!r}."
        )
    if not edge_selector.is_frozen:
        raise ModelFactoryError("edge_selector must be frozen before model construction.")

    # The recipe in the candidate identity and the selector that was actually
    # fitted must agree, otherwise provenance would describe a different S.
    observed = {
        "C": float(edge_selector.config.C),
        "random_state": int(edge_selector.config.random_state),
        "max_iter": int(edge_selector.config.max_iter),
        "input_space": edge_selector.config.input_space,
        "min_edges": int(edge_selector.config.min_edges),
    }
    expected = {
        "C": float(config.C),
        "random_state": int(config.random_state),
        "max_iter": int(config.max_iter),
        "input_space": config.input_space,
        "min_edges": int(config.min_edges),
    }
    if observed != expected:
        raise ModelFactoryError(
            "edge_selector configuration does not match the candidate's "
            f"lasso_edges recipe: selector={observed}, candidate={expected}."
        )

    return SelectedEdgeBackbone(edge_selector, n_rois=int(config.n_rois))


def _build_temporal_aggregator(
    payload: Mapping[str, Any],
    *,
    input_dim: int,
) -> BiGRUAggregator:
    if not isinstance(payload, Mapping):
        raise TypeError("temporal_config must be a mapping.")

    mapping = dict(payload)
    name = mapping.pop("name", None)
    if name != "bigru":
        raise ModelFactoryError(
            "Stage 6K-C currently supports only temporal name='bigru'; "
            f"got {name!r}."
        )

    if "input_dim" in mapping:
        raise ModelFactoryError(
            "BiGRU input_dim is derived from the selected backbone and must "
            "not be stored as an independently tunable candidate parameter."
        )

    # Stage 6H deliberately omits attention-only fields for final/mean
    # pooling. BiGRUConfig owns their scientifically inactive defaults.
    allowed = {
        field.name
        for field in fields(BiGRUConfig)
        if field.name != "input_dim"
    }
    unknown = set(mapping) - allowed
    if unknown:
        raise ModelFactoryError(
            "Unknown BiGRU temporal configuration keys: "
            f"{sorted(unknown)}."
        )

    try:
        config = BiGRUConfig(
            input_dim=int(input_dim),
            **mapping,
        )
    except (TypeError, ValueError) as exc:
        raise ModelFactoryError(
            "Invalid BiGRU temporal configuration."
        ) from exc

    return BiGRUAggregator(config)


def _build_fusion(
    payload: Mapping[str, Any],
    *,
    static_dim: int,
    temporal_dim: int,
) -> ConcatenationFusion:
    if not isinstance(payload, Mapping):
        raise TypeError("fusion_config must be a mapping.")

    mapping = dict(payload)
    name = mapping.pop("name", None)
    if name != "concat":
        raise ModelFactoryError(
            "Stage 6K-C currently supports only fusion name='concat'; "
            f"got {name!r}."
        )

    # Fusion dimensions are consequences of the selected branch models, not
    # independently tunable candidate parameters.
    forbidden = {"static_dim", "temporal_dim", "output_dim"}
    present_forbidden = sorted(forbidden.intersection(mapping))
    if present_forbidden:
        raise ModelFactoryError(
            "Fusion dimensions are derived from the selected branches and "
            "must not be stored as candidate parameters: "
            f"{present_forbidden}."
        )

    if mapping:
        raise ModelFactoryError(
            "Unknown concatenation fusion configuration keys: "
            f"{sorted(mapping)}."
        )

    return ConcatenationFusion(
        static_dim=int(static_dim),
        temporal_dim=int(temporal_dim),
    )


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    dataclass_type: type,
    context: str,
) -> None:
    expected = {field.name for field in fields(dataclass_type)}
    observed = set(payload)

    missing = expected - observed
    unknown = observed - expected

    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise ModelFactoryError(
            f"{context} configuration keys do not match the validated "
            f"runtime schema: {', '.join(details)}."
        )
