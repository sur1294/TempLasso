from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.abide_runtime import (
    ABIDERuntimeConfig,
    ABIDERuntimeFactory,
)
from unified_bn.training.config import (
    LRScheduleConfig,
    OptimizerConfig,
    TrainingConfig,
)
from unified_bn.training.orchestrator import run_validation_search
from unified_bn.training.dataset_protocol import dataset_selection_rule
from unified_bn.training.search_space import (
    ALTERSearchSpace,
    BQNSearchSpace,
    BiGRUSearchSpace,
    BrainNetCNNSearchSpace,
    BNTSearchSpace,
    GCNSearchSpace,
    LassoEdgeSearchSpace,
    generate_fusion_candidates,
    generate_fusion_templasso_candidates,
    generate_templasso_candidates,
    generate_temporal_candidates,
    generate_vanilla_alter_candidates,
    generate_vanilla_bqn_candidates,
    generate_vanilla_candidates,
)


# Resolved ALTER architecture values per named variant. CLI flags default to
# None and override the chosen variant, so "was this flag explicitly set?" is
# simply "is it not None" and the variant itself is the only other knob.
ALTER_CLI_DEFAULTS: dict[str, object] = {
    "alter_layers": 2,
    "alter_dropout": 0.1,
    "alter_pos_encoding": "rrwp",
    "alter_pe_dim": 32,
    "alter_heads": 4,
    "alter_ffn_dim": 1024,
    "alter_cluster_num": 100,
    "alter_edge_threshold": 0.0,
    "alter_transformer_bias": True,
}

# ``paper``   : the released ABIDE configuration (alter/conf/model/lrbgt.yaml),
#               as the paper describes it and as the released code builds on
#               torch 1.x.
# ``harness`` : the ALTER the collaborator's harness actually trains -- the
#               pre-ad5c5a9 checkout with the corr > 0.3 walk-edge rule,
#               nhead hard-coded to 1, and (because it runs on torch >= 2.0
#               with the released positional device/dtype call) no bias in
#               any transformer block. Pinned bit-for-bit in
#               tests/test_alter_harness_variant.py.
ALTER_VARIANTS: dict[str, dict[str, object]] = {
    "paper": {},
    "harness": {
        "alter_heads": 1,
        "alter_edge_threshold": 0.3,
        "alter_transformer_bias": False,
    },
}


def _parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


class ABIDECLIError(RuntimeError):
    """Raised when the real-ABIDE CLI request is internally inconsistent."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unified_bn.training.abide_cli",
        description=(
            "Run one frozen real-ABIDE candidate through the validated "
            "unified training/orchestration stack. This entrypoint performs "
            "train/validation only; held-out test evaluation is intentionally "
            "not exposed here."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("vanilla", "temporal", "fusion", "templasso", "fusion_templasso"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--static-h5", type=Path, required=True)
    parser.add_argument("--temporal-h5", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)

    parser.add_argument("--expected-n-rois", type=int, default=100)
    # Classes in the label column. 2 = the historical binary protocol (ROC-AUC,
    # sensitivity, specificity); K > 2 switches the metrics to macro one-vs-rest
    # ROC-AUC / macro recall / macro specificity and widens every head to K.
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    # The data schema stays shared; this label also selects the Atlas rule.
    parser.add_argument("--dataset-name", default="ABIDE",
                        help="PPMI: max validation macro OvR AUC (4 classes); otherwise min validation loss.")

    # Which static FC encoder the candidate uses. Defaults to 'bqn' so every
    # historical invocation keeps its exact meaning. Applies to every mode
    # that has a static backbone: vanilla (X), temporal (TempX), fusion
    # (X + TempX) and fusion_templasso (X + TempLasso). templasso has none.
    parser.add_argument(
        "--backbone",
        choices=("bqn", "alter", "brainnetcnn", "bnt", "gcn"),
        default="bqn",
    )

    # ALTER architecture. Pick a named variant, then optionally override
    # individual fields. With no flags at all, `--backbone alter` is the
    # released paper configuration.
    parser.add_argument(
        "--alter-variant",
        choices=tuple(ALTER_VARIANTS),
        default="paper",
        help=(
            "paper = released ABIDE configuration; harness = the collaborator's "
            "run (nhead 1, walk edges corr>0.3, bias-free transformer blocks)."
        ),
    )
    parser.add_argument("--alter-layers", type=int, default=None)
    parser.add_argument("--alter-dropout", type=float, default=None)
    parser.add_argument(
        "--alter-pos-encoding",
        choices=("rrwp", "identity", "none"),
        default=None,
    )
    parser.add_argument("--alter-pe-dim", type=int, default=None)
    parser.add_argument("--alter-heads", type=int, default=None)
    parser.add_argument("--alter-ffn-dim", type=int, default=None)
    parser.add_argument("--alter-cluster-num", type=int, default=None)
    parser.add_argument(
        "--alter-edge-threshold",
        type=float,
        default=None,
        help=(
            "FC value above which an entry becomes a random-walk edge. The "
            "released ALTER uses 0.0 (every positive correlation); the "
            "collaborator's harness uses 0.3."
        ),
    )
    parser.add_argument(
        "--alter-transformer-bias",
        type=_parse_bool,
        default=None,
        metavar="{true,false}",
        help=(
            "Bias terms inside the transformer blocks. true = the paper's "
            "model; false = what the released code builds on torch >= 2.0."
        ),
    )

    # BrainNetCNN architecture (models/brainnetcnn/UPSTREAM.md): dropout is
    # the reference's only tuned knob; None means the reference value 0.5.
    parser.add_argument("--bnc-dropout", type=float, default=None)

    # BNT = GraphTransformer from ALTER/models/transformer (see its UPSTREAM.md).
    parser.add_argument("--bnt-layers", type=int, default=None)
    parser.add_argument("--bnt-dropout", type=float, default=None)
    parser.add_argument("--bnt-readout", choices=("mean", "concat", "max", "sum"), default=None)

    # GCN graph construction is fixed; only depth, width and dropout vary.
    parser.add_argument("--gcn-layers", type=int, default=None)
    parser.add_argument("--gcn-hidden-dim", type=int, default=None)
    parser.add_argument("--gcn-dropout", type=float, default=None)

    # BQN architecture: one explicit candidate for smoke/runner use.
    # Required by vanilla/temporal/fusion and enforced per mode below;
    # templasso has no BQN backbone and must not receive them.
    parser.add_argument("--bqn-layers", type=int)
    parser.add_argument(
        "--bqn-activation",
        choices=("gelu", "leaky_relu", "elu"),
    )
    parser.add_argument("--bqn-dropout", type=float)
    parser.add_argument("--bqn-cluster-num", type=int, default=4)

    # TempLasso window encoder: TRAIN-only L1 edge selection. Required by
    # templasso and fusion_templasso; must not be supplied for vanilla /
    # temporal / fusion.
    parser.add_argument("--lasso-C", type=float, dest="lasso_C")
    parser.add_argument(
        "--lasso-input-space",
        choices=("fisher_z", "correlation"),
        default="fisher_z",
    )

    # Training/optimization remain candidate configuration, not hard-coded.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "constant"),
        default="cosine",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)

    # Temporal-branch arguments. These are required by temporal and fusion.
    # Native-K remains the primary Stage-6 path.
    parser.add_argument("--bigru-hidden-dim", type=int)
    parser.add_argument("--bigru-layers", type=int)
    parser.add_argument("--bigru-dropout", type=float, default=0.0)
    parser.add_argument(
        "--bigru-pooling",
        choices=("final", "mean", "attention"),
    )
    parser.add_argument("--attention-hidden-dim", type=int, default=32)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--layer-norm", action="store_true")

    return parser


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        optimizer=OptimizerConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
        ),
        lr_schedule=LRScheduleConfig(
            name=args.lr_schedule,
            min_lr=args.min_lr,
        ),
        checkpoint_selection=dataset_selection_rule(args.dataset_name, args.num_classes),
    )


def build_single_candidate(
    args: argparse.Namespace,
    *,
    split_identity: str,
):
    """
    Materialize exactly one frozen candidate from CLI arguments.

    Stage 6K-C5 remains intentionally a single-candidate real-data entrypoint.
    It is used for end-to-end smoke/runner validation, not full-grid execution.
    Held-out test evaluation is intentionally absent.

    Mode x backbone matrix
    ----------------------
        vanilla           X            (bqn | alter)
        temporal          TempX        (bqn | alter)
        fusion            X + TempX    (bqn | alter, both branches)
        fusion_templasso  X + TempLasso(bqn | alter static branch)
        templasso         TempLasso    (no static backbone)
    """

    training_configs = (build_training_config(args),)

    if args.mode == "templasso":
        _reject_backbone_selection_for_templasso(args)
        _require_temporal_cli_arguments(args)
        _require_lasso_cli_arguments(args)
        _reject_bqn_cli_arguments_for_templasso(args)
        _reject_alter_cli_arguments(args)
        candidates = generate_templasso_candidates(
            seed=args.seed,
            split_identity=split_identity,
            window_policy=WindowPolicyConfig(policy="native"),
            temporal_space=_bigru_space_from_args(args),
            selector_space=LassoEdgeSearchSpace(
                Cs=(args.lasso_C,),
                n_rois=args.expected_n_rois,
                input_space=args.lasso_input_space,
            ),
            training_configs=training_configs,
            num_classes=args.num_classes,
        )
        return _exactly_one(candidates, label="TempLasso")

    # Every remaining mode has a static backbone X. Resolve it once; the
    # helper enforces that only the chosen backbone's flag group is present.
    backbone_space = _backbone_space_from_args(args)

    if args.mode == "vanilla":
        _reject_temporal_cli_arguments_for_vanilla(args)
        _reject_lasso_cli_arguments_for_bqn_modes(args)
        # Backbone-neutral: identical fingerprints to the historical
        # generate_vanilla_{bqn,alter}_candidates calls for their spaces.
        candidates = generate_vanilla_candidates(
            seed=args.seed,
            split_identity=split_identity,
            backbone_space=backbone_space,
            training_configs=training_configs,
            num_classes=args.num_classes,
        )
        return _exactly_one(candidates, label="Vanilla")

    if args.mode == "temporal":
        _require_temporal_cli_arguments(args)
        _reject_lasso_cli_arguments_for_bqn_modes(args)
        candidates = generate_temporal_candidates(
            seed=args.seed,
            split_identity=split_identity,
            window_policy=WindowPolicyConfig(policy="native"),
            temporal_space=_bigru_space_from_args(args),
            backbone_space=backbone_space,
            training_configs=training_configs,
            num_classes=args.num_classes,
        )
        return _exactly_one(candidates, label="Temporal")

    if args.mode == "fusion":
        _require_temporal_cli_arguments(args)
        _reject_lasso_cli_arguments_for_bqn_modes(args)
        candidates = generate_fusion_candidates(
            seed=args.seed,
            split_identity=split_identity,
            window_policy=WindowPolicyConfig(policy="native"),
            temporal_space=_bigru_space_from_args(args),
            backbone_space=backbone_space,
            training_configs=training_configs,
            num_classes=args.num_classes,
        )
        return _exactly_one(candidates, label="Fusion")

    if args.mode == "fusion_templasso":
        # X + TempLasso: static backbone X + lasso_edges temporal branch.
        _require_temporal_cli_arguments(args)
        _require_lasso_cli_arguments(args)
        candidates = generate_fusion_templasso_candidates(
            seed=args.seed,
            split_identity=split_identity,
            window_policy=WindowPolicyConfig(policy="native"),
            temporal_space=_bigru_space_from_args(args),
            backbone_space=backbone_space,
            selector_space=LassoEdgeSearchSpace(
                Cs=(args.lasso_C,),
                n_rois=args.expected_n_rois,
                input_space=args.lasso_input_space,
            ),
            training_configs=training_configs,
            num_classes=args.num_classes,
        )
        return _exactly_one(candidates, label="Fusion-TempLasso")

    raise ABIDECLIError(f"Unsupported mode: {args.mode!r}.")


def _exactly_one(candidates, *, label: str):
    if len(candidates) != 1:
        raise ABIDECLIError(
            f"{label} CLI must materialize exactly one candidate; "
            f"got {len(candidates)}."
        )
    return candidates[0]


def run_from_args(args: argparse.Namespace):
    runtime_factory = ABIDERuntimeFactory(
        ABIDERuntimeConfig(
            split_path=args.split_path,
            static_h5=args.static_h5,
            temporal_h5=args.temporal_h5,
            device=args.device,
            expected_n_rois=args.expected_n_rois,
            pin_memory=args.pin_memory,
            require_stored_split_seed=True,
            dataset_name=args.dataset_name,
            num_classes=args.num_classes,
        )
    )

    if runtime_factory.stored_split_seed != args.seed:
        raise ABIDECLIError(
            "CLI --seed does not match the scalar seed stored in the frozen "
            f"split: cli={args.seed}, split={runtime_factory.stored_split_seed}."
        )

    candidate = build_single_candidate(
        args,
        split_identity=runtime_factory.split_identity,
    )

    selection = run_validation_search(
        (candidate,),
        search_dir=args.search_dir,
        runtime_factory=runtime_factory,
    )
    return selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    selection = run_from_args(args)

    summary = {
        "status": "validation_search_complete",
        "mode": args.mode,
        "dataset": args.dataset_name,
        "backbone": args.backbone,
        "alter_variant": args.alter_variant if args.backbone == "alter" else None,
        "seed": args.seed,
        "search_dir": str(args.search_dir),
        "selected_index": selection.selected_index,
        "selected_fingerprint": selection.selected_fingerprint,
        "selection_metric": selection.selection_metric,
        "selection_mode": selection.selection_mode,
        "selection_value": selection.selection_value,
        "selected_best_checkpoint_path": (
            selection.selected_best_checkpoint_path
        ),
        "selection_json": str(Path(args.search_dir) / "selection.json"),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def resolve_alter_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Named variant first, then any explicitly supplied --alter-* flag."""
    resolved = dict(ALTER_CLI_DEFAULTS)
    resolved.update(ALTER_VARIANTS[args.alter_variant])
    for name in ALTER_CLI_DEFAULTS:
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    return resolved


def _explicit_alter_arguments(args: argparse.Namespace) -> list[str]:
    flags = [
        "--" + name.replace("_", "-")
        for name in ALTER_CLI_DEFAULTS
        if getattr(args, name) is not None
    ]
    if args.alter_variant != "paper":
        flags.insert(0, "--alter-variant")
    return flags


def _alter_space_from_args(args: argparse.Namespace) -> ALTERSearchSpace:
    r = resolve_alter_arguments(args)
    return ALTERSearchSpace(
        n_layers=(r["alter_layers"],),
        dropouts=(r["alter_dropout"],),
        pos_encodings=(r["alter_pos_encoding"],),
        edge_thresholds=(r["alter_edge_threshold"],),
        n_rois=args.expected_n_rois,
        pos_embed_dim=r["alter_pe_dim"],
        nhead=r["alter_heads"],
        ffn_dim=r["alter_ffn_dim"],
        cluster_num=r["alter_cluster_num"],
        transformer_bias=r["alter_transformer_bias"],
    )


def _bqn_space_from_args(args: argparse.Namespace) -> BQNSearchSpace:
    return BQNSearchSpace(
        n_layers=(args.bqn_layers,),
        activations=(args.bqn_activation,),
        dropouts=(args.bqn_dropout,),
        n_rois=args.expected_n_rois,
        cluster_num=args.bqn_cluster_num,
    )


def _backbone_space_from_args(args: argparse.Namespace):
    """Resolve --backbone into a search space, enforcing flag-group exclusivity."""
    if args.backbone == "gcn":
        _reject_bqn_cli_arguments_for_alter(args)
        _reject_alter_cli_arguments(args)
        _reject_bnc_cli_arguments(args)
        _reject_bnt_cli_arguments(args)
        return GCNSearchSpace(
            n_layers=(2 if args.gcn_layers is None else args.gcn_layers,),
            hidden_dims=(64 if args.gcn_hidden_dim is None else args.gcn_hidden_dim,),
            dropouts=(0.1 if args.gcn_dropout is None else args.gcn_dropout,),
            n_rois=args.expected_n_rois,
        )
    _reject_gcn_cli_arguments(args)
    if args.backbone == "bnt":
        _reject_bqn_cli_arguments_for_alter(args)
        _reject_alter_cli_arguments(args)
        _reject_bnc_cli_arguments(args)
        return BNTSearchSpace(
            n_layers=(2 if args.bnt_layers is None else args.bnt_layers,),
            dropouts=(0.1 if args.bnt_dropout is None else args.bnt_dropout,),
            readouts=("mean" if args.bnt_readout is None else args.bnt_readout,),
            n_rois=args.expected_n_rois,
        )
    _reject_bnt_cli_arguments(args)
    if args.backbone == "alter":
        _reject_bqn_cli_arguments_for_alter(args)
        _reject_bnc_cli_arguments(args)
        return _alter_space_from_args(args)
    if args.backbone == "brainnetcnn":
        _reject_bqn_cli_arguments_for_alter(args)
        _reject_alter_cli_arguments(args)
        dropout = 0.5 if args.bnc_dropout is None else args.bnc_dropout
        return BrainNetCNNSearchSpace(dropouts=(dropout,), n_rois=args.expected_n_rois)
    _reject_alter_cli_arguments(args)
    _reject_bnc_cli_arguments(args)
    _require_bqn_cli_arguments(args)
    return _bqn_space_from_args(args)


def _reject_bnc_cli_arguments(args: argparse.Namespace) -> None:
    if getattr(args, "bnc_dropout", None) is not None:
        raise ABIDECLIError(
            f"--backbone {args.backbone} must not define BrainNetCNN arguments: --bnc-dropout"
        )


def _reject_backbone_selection_for_templasso(args: argparse.Namespace) -> None:
    _reject_gcn_cli_arguments(args)
    _reject_bnt_cli_arguments(args)
    if args.backbone != "bqn":
        raise ABIDECLIError(
            "TempLasso has no static backbone; --backbone does not apply to "
            "--mode templasso (use fusion_templasso for X + TempLasso)."
        )


def _reject_gcn_cli_arguments(args: argparse.Namespace) -> None:
    flags = ["--" + name.replace("_", "-") for name in ("gcn_layers", "gcn_hidden_dim", "gcn_dropout")
             if getattr(args, name, None) is not None]
    if flags:
        raise ABIDECLIError(f"--backbone {args.backbone} must not define GCN arguments: {', '.join(flags)}")


def _reject_bnt_cli_arguments(args: argparse.Namespace) -> None:
    flags = ["--" + name.replace("_", "-") for name in ("bnt_layers", "bnt_dropout", "bnt_readout")
             if getattr(args, name, None) is not None]
    if flags:
        raise ABIDECLIError(f"--backbone {args.backbone} must not define BNT arguments: {', '.join(flags)}")


def _reject_alter_cli_arguments(args: argparse.Namespace) -> None:
    explicitly_active = _explicit_alter_arguments(args)
    if explicitly_active:
        raise ABIDECLIError(
            f"--backbone {args.backbone} must not define ALTER arguments: "
            + ", ".join(explicitly_active)
        )


def _reject_bqn_cli_arguments_for_alter(args: argparse.Namespace) -> None:
    explicitly_active = []
    if args.bqn_layers is not None:
        explicitly_active.append("--bqn-layers")
    if args.bqn_activation is not None:
        explicitly_active.append("--bqn-activation")
    if args.bqn_dropout is not None:
        explicitly_active.append("--bqn-dropout")
    if args.bqn_cluster_num != 4:
        explicitly_active.append("--bqn-cluster-num")
    if explicitly_active:
        raise ABIDECLIError(
            f"--backbone {args.backbone} has no BQN backbone and must not define: "
            + ", ".join(explicitly_active)
        )


def _require_temporal_cli_arguments(args: argparse.Namespace) -> None:
    missing = []
    if args.bigru_hidden_dim is None:
        missing.append("--bigru-hidden-dim")
    if args.bigru_layers is None:
        missing.append("--bigru-layers")
    if args.bigru_pooling is None:
        missing.append("--bigru-pooling")

    if missing:
        mode_name = {
            "temporal": "Temporal",
            "fusion": "Fusion",
            "templasso": "TempLasso",
            "fusion_templasso": "Fusion-TempLasso",
        }.get(args.mode, "Temporal")
        raise ABIDECLIError(
            f"{mode_name} mode requires: " + ", ".join(missing)
        )


def _bigru_space_from_args(args: argparse.Namespace) -> BiGRUSearchSpace:
    return BiGRUSearchSpace(
        hidden_dims=(args.bigru_hidden_dim,),
        num_layers=(args.bigru_layers,),
        poolings=(args.bigru_pooling,),
        stacked_dropouts=(args.bigru_dropout,),
        attention_hidden_dims=(args.attention_hidden_dim,),
        attention_dropouts=(args.attention_dropout,),
        projection_dims=(args.projection_dim,),
        layer_norms=(args.layer_norm,),
    )


def _require_bqn_cli_arguments(args: argparse.Namespace) -> None:
    missing = []
    if args.bqn_layers is None:
        missing.append("--bqn-layers")
    if args.bqn_activation is None:
        missing.append("--bqn-activation")
    if args.bqn_dropout is None:
        missing.append("--bqn-dropout")
    if missing:
        raise ABIDECLIError(
            f"{args.mode} mode requires the BQN backbone arguments: "
            + ", ".join(missing)
        )


def _require_lasso_cli_arguments(args: argparse.Namespace) -> None:
    if args.lasso_C is None:
        raise ABIDECLIError("TempLasso mode requires: --lasso-C")


def _reject_bqn_cli_arguments_for_templasso(args: argparse.Namespace) -> None:
    explicitly_active = []
    if args.bqn_layers is not None:
        explicitly_active.append("--bqn-layers")
    if args.bqn_activation is not None:
        explicitly_active.append("--bqn-activation")
    if args.bqn_dropout is not None:
        explicitly_active.append("--bqn-dropout")
    if args.bqn_cluster_num != 4:
        explicitly_active.append("--bqn-cluster-num")
    if explicitly_active:
        raise ABIDECLIError(
            "TempLasso mode has no BQN backbone and must not define: "
            + ", ".join(explicitly_active)
        )


def _reject_lasso_cli_arguments_for_bqn_modes(args: argparse.Namespace) -> None:
    explicitly_active = []
    if args.lasso_C is not None:
        explicitly_active.append("--lasso-C")
    if args.lasso_input_space != "fisher_z":
        explicitly_active.append("--lasso-input-space")
    if explicitly_active:
        raise ABIDECLIError(
            f"{args.mode} mode must not define TempLasso arguments: "
            + ", ".join(explicitly_active)
        )


def _reject_temporal_cli_arguments_for_vanilla(
    args: argparse.Namespace,
) -> None:
    explicitly_active = []

    if args.bigru_hidden_dim is not None:
        explicitly_active.append("--bigru-hidden-dim")
    if args.bigru_layers is not None:
        explicitly_active.append("--bigru-layers")
    if args.bigru_pooling is not None:
        explicitly_active.append("--bigru-pooling")
    if args.bigru_dropout != 0.0:
        explicitly_active.append("--bigru-dropout")
    if args.attention_hidden_dim != 32:
        explicitly_active.append("--attention-hidden-dim")
    if args.attention_dropout != 0.0:
        explicitly_active.append("--attention-dropout")
    if args.projection_dim is not None:
        explicitly_active.append("--projection-dim")
    if args.layer_norm:
        explicitly_active.append("--layer-norm")

    if explicitly_active:
        raise ABIDECLIError(
            "Vanilla mode must not define temporal-model arguments: "
            + ", ".join(explicitly_active)
        )


if __name__ == "__main__":
    raise SystemExit(main())
