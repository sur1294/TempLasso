from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Optional, Sequence, Union

from unified_bn.backbones.alter import ALTERBackboneConfig
from unified_bn.backbones.bqn import BQNBackboneConfig
from unified_bn.backbones.bnt import BNTBackboneConfig
from unified_bn.backbones.gcn import GCNBackboneConfig
from unified_bn.backbones.brainnetcnn import BrainNetCNNBackboneConfig
from unified_bn.backbones.selected_edges import SelectedEdgeBackboneConfig
from unified_bn.temporal.bigru import BiGRUConfig
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.config import (
    CandidateRunConfig,
    RunIdentity,
    TrainingConfig,
)


class SearchSpaceError(ValueError):
    """Raised when a hyperparameter search space is ambiguous or unsafe."""


@dataclass(frozen=True)
class BQNSearchSpace:
    """
    Validation-tuned BQN architecture space.

    Defaults reproduce the current Atlas-style BQN architecture grid:
      depth      {1,2,3,4,5}
      activation {GELU, LeakyReLU, ELU}
      dropout    {0.0,0.1,0.2,0.3}

    The remaining BQN fields are fixed reference-implementation invariants,
    not silently varied hyperparameters.
    """

    n_layers: tuple[int, ...] = (1, 2, 3, 4, 5)
    activations: tuple[str, ...] = ("gelu", "leaky_relu", "elu")
    dropouts: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)

    n_rois: int = 100
    cluster_num: int = 4
    preserve_unused_mlp_b: bool = True
    orthogonal_centers: bool = True
    freeze_centers: bool = True
    project_assignment: bool = True

    def __post_init__(self) -> None:
        _require_tuple_axis("n_layers", self.n_layers)
        _require_tuple_axis("activations", self.activations)
        _require_tuple_axis("dropouts", self.dropouts)

        for value in self.n_layers:
            _require_positive_int("BQN n_layers", value)

        allowed_activations = {"gelu", "leaky_relu", "elu"}
        for value in self.activations:
            if value not in allowed_activations:
                raise SearchSpaceError(
                    "BQN activation must be one of "
                    "{'gelu', 'leaky_relu', 'elu'}; "
                    f"got {value!r}."
                )

        for value in self.dropouts:
            _require_probability("BQN dropout", value)

        _require_positive_int("BQN n_rois", self.n_rois)
        _require_positive_int("BQN cluster_num", self.cluster_num)
        _require_bool("preserve_unused_mlp_b", self.preserve_unused_mlp_b)
        _require_bool("orthogonal_centers", self.orthogonal_centers)
        _require_bool("freeze_centers", self.freeze_centers)
        _require_bool("project_assignment", self.project_assignment)


@dataclass(frozen=True)
class BiGRUSearchSpace:
    """
    Conditional architecture space for the validated BiGRU aggregator.

    This object deliberately has no production tuning defaults for the main
    temporal choices. ``hidden_dims``, ``num_layers`` and ``poolings`` must be
    supplied explicitly when the temporal experiment is frozen.

    ``stacked_dropouts`` means GRU inter-layer dropout candidates. They are
    active only when num_layers > 1. For num_layers == 1, generated candidates
    always use dropout=0.0, matching PyTorch GRU semantics and avoiding
    behaviorally duplicate candidates.

    Attention scorer parameters are varied only when pooling='attention'.
    They are absent from final/mean candidate identity because they are
    behaviorally inactive there.
    """

    hidden_dims: tuple[int, ...]
    num_layers: tuple[int, ...]
    poolings: tuple[str, ...]

    stacked_dropouts: tuple[float, ...] = (0.0,)
    attention_hidden_dims: tuple[int, ...] = (32,)
    attention_dropouts: tuple[float, ...] = (0.0,)
    projection_dims: tuple[int | None, ...] = (None,)
    layer_norms: tuple[bool, ...] = (False,)

    def __post_init__(self) -> None:
        _require_tuple_axis("hidden_dims", self.hidden_dims)
        _require_tuple_axis("num_layers", self.num_layers)
        _require_tuple_axis("poolings", self.poolings)
        _require_tuple_axis("stacked_dropouts", self.stacked_dropouts)
        _require_tuple_axis("attention_hidden_dims", self.attention_hidden_dims)
        _require_tuple_axis("attention_dropouts", self.attention_dropouts)
        _require_tuple_axis("projection_dims", self.projection_dims)
        _require_tuple_axis("layer_norms", self.layer_norms)

        for value in self.hidden_dims:
            _require_positive_int("BiGRU hidden_dim", value)
        for value in self.num_layers:
            _require_positive_int("BiGRU num_layers", value)

        allowed_poolings = {"final", "mean", "attention"}
        for value in self.poolings:
            if value not in allowed_poolings:
                raise SearchSpaceError(
                    "BiGRU pooling must be one of "
                    "{'final', 'mean', 'attention'}; "
                    f"got {value!r}."
                )

        for value in self.stacked_dropouts:
            _require_probability("BiGRU stacked dropout", value)
        for value in self.attention_hidden_dims:
            _require_positive_int("BiGRU attention_hidden_dim", value)
        for value in self.attention_dropouts:
            _require_probability("BiGRU attention dropout", value)

        for value in self.projection_dims:
            if value is not None:
                _require_positive_int("BiGRU projection_dim", value)

        for value in self.layer_norms:
            _require_bool("BiGRU layer_norm", value)

        if all(value == 1 for value in self.num_layers):
            if any(float(value) != 0.0 for value in self.stacked_dropouts):
                raise SearchSpaceError(
                    "stacked_dropouts contains nonzero values but every "
                    "BiGRU candidate has num_layers=1. PyTorch GRU dropout is "
                    "inactive for a one-layer GRU; use stacked_dropouts=(0.0,)."
                )


def generate_bqn_backbone_configs(
    space: Optional[BQNSearchSpace] = None,
) -> tuple[dict[str, Any], ...]:
    """
    Expand a BQN architecture space into deterministic configuration mappings.

    The returned mapping carries ``name='bqn'`` for future model-factory
    dispatch. All remaining fields match ``BQNBackboneConfig`` exactly.
    No model is constructed here.
    """

    if space is None:
        space = BQNSearchSpace()
    if not isinstance(space, BQNSearchSpace):
        raise TypeError(
            f"space must be BQNSearchSpace, got {type(space)!r}."
        )

    configs: list[dict[str, Any]] = []
    for n_layers, activation, dropout in product(
        space.n_layers,
        space.activations,
        space.dropouts,
    ):
        runtime_config = BQNBackboneConfig(
            n_rois=space.n_rois,
            n_layers=n_layers,
            dropout=float(dropout),
            activation=activation,
            cluster_num=space.cluster_num,
            preserve_unused_mlp_b=space.preserve_unused_mlp_b,
            orthogonal_centers=space.orthogonal_centers,
            freeze_centers=space.freeze_centers,
            project_assignment=space.project_assignment,
        )
        payload = {"name": "bqn", **asdict(runtime_config)}
        configs.append(payload)

    _require_unique_mappings(configs, context="BQN architecture")
    return tuple(configs)


def generate_bigru_temporal_configs(
    space: BiGRUSearchSpace,
) -> tuple[dict[str, Any], ...]:
    """
    Expand the validated BiGRU architecture with conditional hyperparameters.

    Generated mappings intentionally omit ``input_dim`` because it is derived
    from the selected backbone at model-construction time; it is not a
    tunable temporal hyperparameter.

    For final/mean pooling, attention-only parameters are omitted from
    candidate identity. For attention pooling they are explicit.
    """

    if not isinstance(space, BiGRUSearchSpace):
        raise TypeError(
            f"space must be BiGRUSearchSpace, got {type(space)!r}."
        )

    configs: list[dict[str, Any]] = []

    for hidden_dim, num_layers, pooling in product(
        space.hidden_dims,
        space.num_layers,
        space.poolings,
    ):
        gru_dropouts = (
            (0.0,)
            if num_layers == 1
            else space.stacked_dropouts
        )

        if pooling == "attention":
            attention_axes = tuple(
                product(
                    space.attention_hidden_dims,
                    space.attention_dropouts,
                )
            )
        else:
            attention_axes = ((None, None),)

        for (
            dropout,
            attention_pair,
            projection_dim,
            layer_norm,
        ) in product(
            gru_dropouts,
            attention_axes,
            space.projection_dims,
            space.layer_norms,
        ):
            attention_hidden_dim, attention_dropout = attention_pair

            runtime_config = BiGRUConfig(
                input_dim=1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=float(dropout),
                pooling=pooling,
                attention_hidden_dim=(
                    32
                    if attention_hidden_dim is None
                    else attention_hidden_dim
                ),
                attention_dropout=(
                    0.0
                    if attention_dropout is None
                    else float(attention_dropout)
                ),
                projection_dim=projection_dim,
                layer_norm=layer_norm,
            )

            raw = asdict(runtime_config)
            raw.pop("input_dim")

            if pooling != "attention":
                raw.pop("attention_hidden_dim")
                raw.pop("attention_dropout")

            payload = {"name": "bigru", **raw}
            configs.append(payload)

    _require_unique_mappings(configs, context="BiGRU architecture")
    return tuple(configs)


def generate_vanilla_bqn_candidates(
    *,
    seed: int,
    split_identity: str,
    backbone_space: Optional[BQNSearchSpace] = None,
    training_configs: Sequence[TrainingConfig] = (TrainingConfig(),),
    num_classes: int = 2,
) -> tuple[CandidateRunConfig, ...]:
    """
    Build validation-tuning candidates for the Vanilla+BQN pipeline.

    This function only materializes configuration objects. It has no dataset,
    DataLoader, model, optimizer, trainer, result, or test-loader access.
    """

    if backbone_space is None:
        backbone_space = BQNSearchSpace()

    training_options = _validated_training_configs(training_configs)
    candidates: list[CandidateRunConfig] = []

    for backbone_config, training_config in product(
        generate_bqn_backbone_configs(backbone_space),
        training_options,
    ):
        candidates.append(
            CandidateRunConfig(
                identity=RunIdentity(
                    num_classes=num_classes,
                    mode="vanilla",
                    seed=seed,
                    split_identity=split_identity,
                    backbone_config=backbone_config,
                ),
                training=training_config,
            )
        )

    _require_unique_candidates(candidates)
    return tuple(candidates)


@dataclass(frozen=True)
class ALTERSearchSpace:
    """
    Validation-tuned ALTER architecture space.

    Defaults are a SINGLETON reproducing the released ALTER ABIDE
    configuration (``alter/conf/model/lrbgt.yaml``): two encoder blocks, RRWP
    long-range encoding of walk length 32, torch's 0.1 transformer dropout,
    and the upstream ``corr > 0`` walk-edge rule.

    Unlike ``BQNSearchSpace`` the defaults do not span a grid. ALTER is being
    introduced as an external baseline, and a baseline whose hyperparameters
    we tuned but whose authors' were not is not a fair comparison. Widening
    any axis is therefore an explicit, visible decision at the call site.

    The axes that exist are the ones with an open scientific question:

    ``pos_encodings``
        ALTER's central claim is that the long-range random-walk encoding is
        what helps, so ``('rrwp',)`` vs ``('rrwp', 'identity', 'none')`` is
        the natural ablation.
    ``edge_thresholds``
        The released code walks over ``corr > 0``. The collaborator's own
        harness comments describe a 0.3 threshold "same as ALTER", which the
        released code does not implement. Keeping this an axis means settling
        that question is a flag, not a patch.

    The remaining fields are fixed reference-implementation invariants, not
    silently varied hyperparameters.
    """

    n_layers: tuple[int, ...] = (2,)
    dropouts: tuple[float, ...] = (0.1,)
    pos_encodings: tuple[str, ...] = ("rrwp",)
    edge_thresholds: tuple[float, ...] = (0.0,)

    n_rois: int = 100
    pos_embed_dim: int = 32
    nhead: int = 4
    ffn_dim: int = 1024
    cluster_num: int = 100
    # See ALTERConfig.transformer_bias. False reproduces the collaborator's
    # harness (released code on torch >= 2.0); True is the paper's model.
    transformer_bias: bool = True
    orthogonal_centers: bool = True
    freeze_centers: bool = True
    project_assignment: bool = True

    def __post_init__(self) -> None:
        _require_tuple_axis("n_layers", self.n_layers)
        _require_tuple_axis("dropouts", self.dropouts)
        _require_tuple_axis("pos_encodings", self.pos_encodings)
        _require_tuple_axis("edge_thresholds", self.edge_thresholds)
        _require_bool("transformer_bias", self.transformer_bias)

        for value in self.n_layers:
            _require_positive_int("ALTER n_layers", value)
        for value in self.dropouts:
            _require_probability("ALTER dropout", value)

        allowed_encodings = {"rrwp", "identity", "none"}
        for value in self.pos_encodings:
            if value not in allowed_encodings:
                raise SearchSpaceError(
                    "ALTER pos_encoding must be one of "
                    "{'rrwp', 'identity', 'none'}; "
                    f"got {value!r}."
                )

        for value in self.edge_thresholds:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SearchSpaceError(
                    "ALTER edge_threshold values must be numbers."
                )
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise SearchSpaceError(
                    "ALTER edge_threshold values must be finite and "
                    "nonnegative."
                )

        _require_positive_int("ALTER n_rois", self.n_rois)
        _require_positive_int("ALTER pos_embed_dim", self.pos_embed_dim)
        _require_positive_int("ALTER nhead", self.nhead)
        _require_positive_int("ALTER ffn_dim", self.ffn_dim)
        _require_positive_int("ALTER cluster_num", self.cluster_num)
        _require_bool("orthogonal_centers", self.orthogonal_centers)
        _require_bool("freeze_centers", self.freeze_centers)
        _require_bool("project_assignment", self.project_assignment)


def generate_alter_backbone_configs(
    space: Optional[ALTERSearchSpace] = None,
) -> tuple[dict[str, Any], ...]:
    """
    Expand an ALTER architecture space into deterministic configuration
    mappings.

    The returned mapping carries ``name='alter'`` for model-factory dispatch.
    All remaining fields match ``ALTERBackboneConfig`` exactly, whose
    ``__post_init__`` delegates to the vendored ``ALTERConfig`` contract -- so
    an architecturally impossible combination (for instance a ``d_model`` that
    is not divisible by ``nhead``) is rejected here, before any candidate is
    ever scheduled. No model is constructed.
    """

    if space is None:
        space = ALTERSearchSpace()
    if not isinstance(space, ALTERSearchSpace):
        raise TypeError(
            f"space must be ALTERSearchSpace, got {type(space)!r}."
        )

    configs: list[dict[str, Any]] = []
    for n_layers, dropout, pos_encoding, edge_threshold in product(
        space.n_layers,
        space.dropouts,
        space.pos_encodings,
        space.edge_thresholds,
    ):
        try:
            runtime_config = ALTERBackboneConfig(
                n_rois=space.n_rois,
                n_layers=n_layers,
                pos_encoding=pos_encoding,
                pos_embed_dim=space.pos_embed_dim,
                nhead=space.nhead,
                ffn_dim=space.ffn_dim,
                dropout=float(dropout),
                cluster_num=space.cluster_num,
                edge_threshold=float(edge_threshold),
                transformer_bias=space.transformer_bias,
                orthogonal_centers=space.orthogonal_centers,
                freeze_centers=space.freeze_centers,
                project_assignment=space.project_assignment,
            )
        except (TypeError, ValueError) as exc:
            raise SearchSpaceError(
                "ALTER search space contains an unbuildable architecture: "
                f"{exc}"
            ) from exc
        payload = {"name": "alter", **asdict(runtime_config)}
        configs.append(payload)

    _require_unique_mappings(configs, context="ALTER architecture")
    return tuple(configs)


@dataclass(frozen=True)
class BrainNetCNNSearchSpace:
    """
    Validation-tuned BrainNetCNN space (Kawahara et al. 2017, transcribed
    from the collaborator's harness; see ``models/brainnetcnn/UPSTREAM.md``).

    The reference architecture has a single regularisation knob the original
    authors tuned (dropout), so that is the only axis; widths are fixed
    reference invariants. Defaults are a singleton reproducing the reference
    (dropout 0.5).
    """

    dropouts: tuple[float, ...] = (0.5,)

    n_rois: int = 100
    e2e_channels: tuple[int, int] = (32, 64)
    n2g_channels: int = 256
    dense_dims: tuple[int, int] = (128, 30)
    negative_slope: float = 0.33
    final_activation: bool = True

    def __post_init__(self) -> None:
        _require_tuple_axis("dropouts", self.dropouts)
        for value in self.dropouts:
            _require_probability("BrainNetCNN dropout", value)
        _require_positive_int("BrainNetCNN n_rois", self.n_rois)
        _require_positive_int("BrainNetCNN n2g_channels", self.n2g_channels)
        _require_bool("final_activation", self.final_activation)
        if len(self.e2e_channels) != 2 or len(self.dense_dims) != 2:
            raise SearchSpaceError(
                "BrainNetCNN e2e_channels and dense_dims must each have two entries."
            )
        for value in (*self.e2e_channels, *self.dense_dims):
            _require_positive_int("BrainNetCNN width", value)


def generate_brainnetcnn_backbone_configs(
    space: Optional[BrainNetCNNSearchSpace] = None,
) -> tuple[dict[str, Any], ...]:
    """Expand a BrainNetCNN space into ``name='brainnetcnn'`` mappings."""
    if space is None:
        space = BrainNetCNNSearchSpace()
    if not isinstance(space, BrainNetCNNSearchSpace):
        raise TypeError(
            f"space must be BrainNetCNNSearchSpace, got {type(space)!r}."
        )
    configs: list[dict[str, Any]] = []
    for dropout in space.dropouts:
        try:
            runtime_config = BrainNetCNNBackboneConfig(
                n_rois=space.n_rois,
                e2e_channels=tuple(space.e2e_channels),
                n2g_channels=space.n2g_channels,
                dense_dims=tuple(space.dense_dims),
                dropout=float(dropout),
                negative_slope=float(space.negative_slope),
                final_activation=space.final_activation,
            )
        except (TypeError, ValueError) as exc:
            raise SearchSpaceError(
                "BrainNetCNN search space contains an unbuildable architecture: "
                f"{exc}"
            ) from exc
        payload = {"name": "brainnetcnn", **asdict(runtime_config)}
        # JSON identity: tuples become lists in every serialised form.
        payload["e2e_channels"] = list(payload["e2e_channels"])
        payload["dense_dims"] = list(payload["dense_dims"])
        configs.append(payload)
    _require_unique_mappings(configs, context="BrainNetCNN architecture")
    return tuple(configs)


@dataclass(frozen=True)
class BNTSearchSpace:
    """The supplied GraphTransformer; defaults reproduce its mean-readout run.

    Only layers, dropout, and readout are search axes. Every combination is
    validated before constructing a model. No data or outcomes are used here.
    """
    n_layers: tuple[int, ...] = (2,)
    dropouts: tuple[float, ...] = (0.1,)
    readouts: tuple[str, ...] = ("mean",)
    n_rois: int = 100
    nhead: int = 4
    ffn_dim: int = 1024

    def __post_init__(self):
        for name in ("n_layers", "dropouts", "readouts"):
            _require_tuple_axis(name, getattr(self, name))
        for layers, dropout, readout in product(self.n_layers, self.dropouts, self.readouts):
            try:
                BNTBackboneConfig(n_rois=self.n_rois, n_layers=layers, dropout=dropout,
                                  readout=readout, nhead=self.nhead, ffn_dim=self.ffn_dim)
            except (TypeError, ValueError) as exc:
                raise SearchSpaceError(f"Invalid BNT search space: {exc}") from exc


def generate_bnt_backbone_configs(space: Optional[BNTSearchSpace] = None):
    space = BNTSearchSpace() if space is None else space
    if not isinstance(space, BNTSearchSpace):
        raise TypeError("space must be BNTSearchSpace.")
    configs = []
    for layers, dropout, readout in product(space.n_layers, space.dropouts, space.readouts):
        config = BNTBackboneConfig(n_rois=space.n_rois, n_layers=layers, dropout=float(dropout),
                                  readout=readout, nhead=space.nhead, ffn_dim=space.ffn_dim)
        configs.append({"name": "bnt", **asdict(config)})
    _require_unique_mappings(configs, context="BNT architecture")
    return tuple(configs)


@dataclass(frozen=True)
class GCNSearchSpace:
    """Explicit GCN axes; graph construction and spatial readout stay fixed."""

    n_layers: tuple[int, ...] = (2,)
    hidden_dims: tuple[int, ...] = (64,)
    dropouts: tuple[float, ...] = (0.1,)
    n_rois: int = 100
    adjacency: str = "positive"
    node_features: str = "fc_rows"
    readout: str = "mean"

    def __post_init__(self):
        for name in ("n_layers", "hidden_dims", "dropouts"):
            _require_tuple_axis(name, getattr(self, name))
        for layers, width, dropout in product(self.n_layers, self.hidden_dims, self.dropouts):
            try:
                GCNBackboneConfig(n_rois=self.n_rois, n_layers=layers, hidden_dim=width,
                                  dropout=dropout, adjacency=self.adjacency,
                                  node_features=self.node_features, readout=self.readout)
            except (TypeError, ValueError) as exc:
                raise SearchSpaceError(f"Invalid GCN search space: {exc}") from exc


def generate_gcn_backbone_configs(space: Optional[GCNSearchSpace] = None):
    space = GCNSearchSpace() if space is None else space
    if not isinstance(space, GCNSearchSpace):
        raise TypeError("space must be GCNSearchSpace.")
    configs = []
    for layers, width, dropout in product(space.n_layers, space.hidden_dims, space.dropouts):
        config = GCNBackboneConfig(n_rois=space.n_rois, n_layers=layers, hidden_dim=width,
                                  dropout=float(dropout), adjacency=space.adjacency,
                                  node_features=space.node_features, readout=space.readout)
        configs.append({"name": "gcn", **asdict(config)})
    _require_unique_mappings(configs, context="GCN architecture")
    return tuple(configs)


BackboneSearchSpace = Union[BQNSearchSpace, ALTERSearchSpace, BrainNetCNNSearchSpace, BNTSearchSpace, GCNSearchSpace]
"""
Any static FC-encoder search space.

The temporal and fusion generators below are written against this union
rather than against ``BQNSearchSpace`` because the pipelines they feed are
backbone-agnostic: ``TemporalPipeline`` and ``FusionPipeline`` only require a
``BaseBackbone``, and ``model_factory`` dispatches on ``backbone_config["name"]``.
Typing the generators to BQN would have made TempX / X+TempX / X+TempLasso
BQN-only for no scientific reason.
"""


def generate_backbone_configs(
    space: BackboneSearchSpace,
) -> tuple[dict[str, Any], ...]:
    """Expand any registered backbone space into ``name``-tagged mappings."""
    if isinstance(space, BQNSearchSpace):
        return generate_bqn_backbone_configs(space)
    if isinstance(space, ALTERSearchSpace):
        return generate_alter_backbone_configs(space)
    if isinstance(space, BrainNetCNNSearchSpace):
        return generate_brainnetcnn_backbone_configs(space)
    if isinstance(space, BNTSearchSpace):
        return generate_bnt_backbone_configs(space)
    if isinstance(space, GCNSearchSpace):
        return generate_gcn_backbone_configs(space)
    raise TypeError(
        "backbone_space must be BQNSearchSpace, ALTERSearchSpace or "
        f"BrainNetCNNSearchSpace, BNTSearchSpace or GCNSearchSpace, got {type(space)!r}."
    )


def generate_vanilla_candidates(
    *,
    seed: int,
    split_identity: str,
    backbone_space: BackboneSearchSpace,
    training_configs: Sequence[TrainingConfig] = (TrainingConfig(),),
    num_classes: int = 2,
) -> tuple[CandidateRunConfig, ...]:
    """
    Backbone-neutral vanilla (X) candidate generator: the static backbone
    grid times the training grid, identical in structure to the BQN and
    ALTER generators (and equal to them, fingerprint for fingerprint, for
    their own spaces).
    """
    training_options = _validated_training_configs(training_configs)
    candidates: list[CandidateRunConfig] = []
    for backbone_config, training_config in product(
        generate_backbone_configs(backbone_space),
        training_options,
    ):
        candidates.append(
            CandidateRunConfig(
                identity=RunIdentity(
                    num_classes=num_classes,
                    mode="vanilla",
                    seed=seed,
                    split_identity=split_identity,
                    backbone_config=backbone_config,
                ),
                training=training_config,
            )
        )
    _require_unique_candidates(candidates)
    return tuple(candidates)


def generate_vanilla_alter_candidates(
    *,
    seed: int,
    split_identity: str,
    backbone_space: Optional[ALTERSearchSpace] = None,
    training_configs: Sequence[TrainingConfig] = (TrainingConfig(),),
    num_classes: int = 2,
) -> tuple[CandidateRunConfig, ...]:
    """
    Build validation-tuning candidates for the Vanilla+ALTER pipeline.

    Structurally identical to :func:`generate_vanilla_bqn_candidates`; only
    ``backbone_config`` differs, which is the point -- the two static
    baselines then share the trainer, the optimiser, the schedule, the
    checkpoint rule and the frozen split, so a difference in their numbers is
    a difference between the models.

    This function only materializes configuration objects. It has no dataset,
    DataLoader, model, optimizer, trainer, result, or test-loader access.
    """

    if backbone_space is None:
        backbone_space = ALTERSearchSpace()

    training_options = _validated_training_configs(training_configs)
    candidates: list[CandidateRunConfig] = []

    for backbone_config, training_config in product(
        generate_alter_backbone_configs(backbone_space),
        training_options,
    ):
        candidates.append(
            CandidateRunConfig(
                identity=RunIdentity(
                    num_classes=num_classes,
                    mode="vanilla",
                    seed=seed,
                    split_identity=split_identity,
                    backbone_config=backbone_config,
                ),
                training=training_config,
            )
        )

    _require_unique_candidates(candidates)
    return tuple(candidates)


def generate_temporal_bqn_bigru_candidates(
    *,
    seed: int,
    split_identity: str,
    window_policy: WindowPolicyConfig,
    temporal_space: BiGRUSearchSpace,
    backbone_space: Optional[BackboneSearchSpace] = None,
    training_configs: Sequence[TrainingConfig] = (TrainingConfig(),),
    num_classes: int = 2,
) -> tuple[CandidateRunConfig, ...]:
    """
    Build validation-tuning candidates for TempX: shared static backbone X
    applied to every dynamic window, then BiGRU.

    Historically BQN-only (hence the name, kept for compatibility);
    ``backbone_space`` now accepts any registered backbone space, so TempALTER
    is the same call with an ``ALTERSearchSpace``. The backbone and BiGRU
    spaces are independent. A caller may therefore pass:
      - a singleton backbone space to hold the architecture fixed, or
      - a broader backbone space for joint backbone+temporal tuning.

    The choice is an experiment-level decision; this module does not silently
    impose one. Held-out test data is absent from this API.
    """

    if not isinstance(window_policy, WindowPolicyConfig):
        raise TypeError(
            "window_policy must be WindowPolicyConfig, got "
            f"{type(window_policy)!r}."
        )
    if backbone_space is None:
        backbone_space = BQNSearchSpace()

    training_options = _validated_training_configs(training_configs)
    backbone_options = generate_backbone_configs(backbone_space)
    temporal_options = generate_bigru_temporal_configs(temporal_space)

    candidates: list[CandidateRunConfig] = []
    for backbone_config, temporal_config, training_config in product(
        backbone_options,
        temporal_options,
        training_options,
    ):
        candidates.append(
            CandidateRunConfig(
                identity=RunIdentity(
                    num_classes=num_classes,
                    mode="temporal",
                    seed=seed,
                    split_identity=split_identity,
                    backbone_config=backbone_config,
                    temporal_config=temporal_config,
                    window_policy=window_policy,
                ),
                training=training_config,
            )
        )

    _require_unique_candidates(candidates)
    return tuple(candidates)


# Backbone-neutral name for the same generator.
generate_temporal_candidates = generate_temporal_bqn_bigru_candidates


def generate_fusion_candidates(
    *,
    seed: int,
    split_identity: str,
    window_policy: WindowPolicyConfig,
    temporal_space: BiGRUSearchSpace,
    backbone_space: Optional[BackboneSearchSpace] = None,
    training_configs: Sequence[TrainingConfig] = (TrainingConfig(),),
    num_classes: int = 2,
) -> tuple[CandidateRunConfig, ...]:
    """
    Build validation-tuning candidates for X + TempX (simple late fusion).

        static FC  -> X            -> h_static  \
                                                 concat -> classifier
        windows    -> X (shared)   -> BiGRU -> h_dyn /

    Both branches are built from the same ``backbone_config`` but are
    DISTINCT module instances with independent weights; that is the
    historical controlled baseline and ``temporal_backbone_config`` is left
    unset so existing X+TempX fingerprints are unchanged. Previously this
    candidate was assembled inline in the CLI; it lives here so that any
    registered backbone, not only BQN, gets the same treatment. Held-out test
    data is absent from this API.
    """

    if not isinstance(window_policy, WindowPolicyConfig):
        raise TypeError(
            "window_policy must be WindowPolicyConfig, got "
            f"{type(window_policy)!r}."
        )
    if backbone_space is None:
        backbone_space = BQNSearchSpace()

    training_options = _validated_training_configs(training_configs)
    backbone_options = generate_backbone_configs(backbone_space)
    temporal_options = generate_bigru_temporal_configs(temporal_space)

    candidates: list[CandidateRunConfig] = []
    for backbone_config, temporal_config, training_config in product(
        backbone_options,
        temporal_options,
        training_options,
    ):
        candidates.append(
            CandidateRunConfig(
                identity=RunIdentity(
                    num_classes=num_classes,
                    mode="fusion",
                    seed=seed,
                    split_identity=split_identity,
                    backbone_config=backbone_config,
                    temporal_config=temporal_config,
                    window_policy=window_policy,
                    fusion_config={"name": "concat"},
                ),
                training=training_config,
            )
        )

    _require_unique_candidates(candidates)
    return tuple(candidates)


@dataclass(frozen=True)
class LassoEdgeSearchSpace:
    """
    Validation-tuned space for the TempLasso window encoder.

    Only ``Cs`` is a genuine hyperparameter: it controls how many edges
    survive L1 selection. The remaining fields are fixed recipe invariants
    carried into the candidate identity so the runtime can re-fit the same S
    for each seed's TRAIN partition.
    """

    Cs: tuple[float, ...] = (0.005, 0.01, 0.03)
    n_rois: int = 100
    random_state: int = 0
    max_iter: int = 3000
    input_space: str = "fisher_z"
    min_edges: int = 5

    def __post_init__(self) -> None:
        _require_tuple_axis("Cs", self.Cs)
        for value in self.Cs:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SearchSpaceError("lasso_edges C values must be numbers.")
            if value <= 0.0:
                raise SearchSpaceError("lasso_edges C values must be positive.")
        _require_positive_int("lasso_edges n_rois", self.n_rois)
        if self.n_rois <= 1:
            raise SearchSpaceError("lasso_edges n_rois must be > 1.")
        if self.input_space not in ("fisher_z", "correlation"):
            raise SearchSpaceError(
                "lasso_edges input_space must be 'fisher_z' or 'correlation'."
            )


def generate_lasso_edge_backbone_configs(
    space: Optional[LassoEdgeSearchSpace] = None,
) -> tuple[dict[str, Any], ...]:
    """
    Expand a LassoEdgeSearchSpace into ``name='lasso_edges'`` recipes.

    No data is touched and no edge set is materialised here.
    """

    if space is None:
        space = LassoEdgeSearchSpace()
    if not isinstance(space, LassoEdgeSearchSpace):
        raise TypeError(
            f"space must be LassoEdgeSearchSpace, got {type(space)!r}."
        )

    configs: list[dict[str, Any]] = []
    for C in space.Cs:
        runtime_config = SelectedEdgeBackboneConfig(
            n_rois=space.n_rois,
            C=float(C),
            random_state=space.random_state,
            max_iter=space.max_iter,
            input_space=space.input_space,  # type: ignore[arg-type]
            min_edges=space.min_edges,
        )
        configs.append({"name": "lasso_edges", **asdict(runtime_config)})

    _require_unique_mappings(configs, context="lasso_edges recipe")
    return tuple(configs)


def generate_templasso_candidates(
    *,
    seed: int,
    split_identity: str,
    window_policy: WindowPolicyConfig,
    temporal_space: BiGRUSearchSpace,
    selector_space: Optional[LassoEdgeSearchSpace] = None,
    training_configs: Sequence[TrainingConfig] = (TrainingConfig(),),
    num_classes: int = 2,
) -> tuple[CandidateRunConfig, ...]:
    """
    Build validation-tuning candidates for TempLasso.

        (K_i, E) --[S fit on TRAIN]--> (K_i, |S|) --> BiGRU --> classifier

    These are ordinary ``mode='temporal'`` candidates whose backbone is the
    parameter-free ``lasso_edges`` window encoder, so the generic trainer,
    orchestrator, checkpointing and validation-only selection apply
    unchanged. Held-out test data is absent from this API.
    """

    if not isinstance(window_policy, WindowPolicyConfig):
        raise TypeError(
            "window_policy must be WindowPolicyConfig, got "
            f"{type(window_policy)!r}."
        )
    if selector_space is None:
        selector_space = LassoEdgeSearchSpace()

    training_options = _validated_training_configs(training_configs)
    backbone_options = generate_lasso_edge_backbone_configs(selector_space)
    temporal_options = generate_bigru_temporal_configs(temporal_space)

    candidates: list[CandidateRunConfig] = []
    for backbone_config, temporal_config, training_config in product(
        backbone_options,
        temporal_options,
        training_options,
    ):
        candidates.append(
            CandidateRunConfig(
                identity=RunIdentity(
                    num_classes=num_classes,
                    mode="temporal",
                    seed=seed,
                    split_identity=split_identity,
                    backbone_config=backbone_config,
                    temporal_config=temporal_config,
                    window_policy=window_policy,
                ),
                training=training_config,
            )
        )

    _require_unique_candidates(candidates)
    return tuple(candidates)


def generate_fusion_templasso_candidates(
    *,
    seed: int,
    split_identity: str,
    window_policy: WindowPolicyConfig,
    temporal_space: BiGRUSearchSpace,
    backbone_space: Optional[BackboneSearchSpace] = None,
    selector_space: Optional[LassoEdgeSearchSpace] = None,
    training_configs: Sequence[TrainingConfig] = (TrainingConfig(),),
    num_classes: int = 2,
) -> tuple[CandidateRunConfig, ...]:
    """
    Build validation-tuning candidates for X + TempLasso (late fusion).

        static FC  -> X              -> h_static  \
                                                    concat -> classifier
        windows    -> lasso_edges S  -> BiGRU -> h_dyn /

    ``backbone_config`` carries the static branch (any registered backbone)
    and ``temporal_backbone_config`` the lasso_edges window encoder; the two
    are independent axes, so a singleton backbone space holds the static
    branch fixed while C / BiGRU are tuned, or vice versa. Held-out test data
    is absent.
    """

    if not isinstance(window_policy, WindowPolicyConfig):
        raise TypeError(
            "window_policy must be WindowPolicyConfig, got "
            f"{type(window_policy)!r}."
        )
    if backbone_space is None:
        backbone_space = BQNSearchSpace()
    if selector_space is None:
        selector_space = LassoEdgeSearchSpace()
    if selector_space.n_rois != backbone_space.n_rois:
        raise SearchSpaceError(
            "Static backbone and lasso_edges branches must share n_rois: "
            f"{backbone_space.n_rois} vs {selector_space.n_rois}."
        )

    training_options = _validated_training_configs(training_configs)
    static_options = generate_backbone_configs(backbone_space)
    selector_options = generate_lasso_edge_backbone_configs(selector_space)
    temporal_options = generate_bigru_temporal_configs(temporal_space)

    candidates: list[CandidateRunConfig] = []
    for static_config, selector_config, temporal_config, training_config in product(
        static_options, selector_options, temporal_options, training_options,
    ):
        candidates.append(
            CandidateRunConfig(
                identity=RunIdentity(
                    num_classes=num_classes,
                    mode="fusion",
                    seed=seed,
                    split_identity=split_identity,
                    backbone_config=static_config,
                    temporal_config=temporal_config,
                    window_policy=window_policy,
                    fusion_config={"name": "concat"},
                    temporal_backbone_config=selector_config,
                ),
                training=training_config,
            )
        )

    _require_unique_candidates(candidates)
    return tuple(candidates)


def candidate_fingerprint(config: CandidateRunConfig) -> str:
    """
    Return a stable SHA-256 identity for one complete candidate configuration.

    Seed, semantic split identity, architecture, window policy, and training
    configuration are all included through ``CandidateRunConfig.to_dict()``.
    """

    if not isinstance(config, CandidateRunConfig):
        raise TypeError(
            "config must be CandidateRunConfig, got "
            f"{type(config)!r}."
        )

    encoded = _canonical_json(config.to_dict()).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_training_configs(
    configs: Sequence[TrainingConfig],
) -> tuple[TrainingConfig, ...]:
    if isinstance(configs, (str, bytes)) or not isinstance(configs, Sequence):
        raise TypeError("training_configs must be a finite sequence.")

    values = tuple(configs)
    if not values:
        raise SearchSpaceError("training_configs must be non-empty.")

    for value in values:
        if not isinstance(value, TrainingConfig):
            raise TypeError(
                "Every training_configs entry must be TrainingConfig; "
                f"got {type(value)!r}."
            )

    payloads = [value.to_dict() for value in values]
    _require_unique_mappings(payloads, context="training configuration")
    return values


def _require_unique_candidates(
    candidates: Sequence[CandidateRunConfig],
) -> None:
    fingerprints = [candidate_fingerprint(value) for value in candidates]
    if len(fingerprints) != len(set(fingerprints)):
        raise SearchSpaceError(
            "Candidate generation produced duplicate complete configurations."
        )


def _require_unique_mappings(
    mappings: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    encoded = [_canonical_json(value) for value in mappings]
    if len(encoded) != len(set(encoded)):
        raise SearchSpaceError(
            f"{context} search space produced duplicate configurations."
        )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SearchSpaceError(
            "Candidate configuration must be finite and JSON serializable."
        ) from exc


def _require_tuple_axis(name: str, values: tuple[Any, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError(
            f"{name} must be a tuple so the search definition is immutable."
        )
    if not values:
        raise SearchSpaceError(f"{name} must be non-empty.")

    canonical_values = [_canonical_scalar(value) for value in values]
    if len(canonical_values) != len(set(canonical_values)):
        raise SearchSpaceError(
            f"{name} contains duplicate search values: {values!r}."
        )


def _canonical_scalar(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SearchSpaceError(
            f"Search-axis value is not finite/JSON-safe: {value!r}."
        ) from exc


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise SearchSpaceError(f"{name} must be positive.")


def _require_probability(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric < 1.0:
        raise SearchSpaceError(
            f"{name} must satisfy 0 <= value < 1."
        )


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
