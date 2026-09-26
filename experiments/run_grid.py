#!/usr/bin/env python3
"""
Validation-only hyperparameter search for one table row, parallelisable
across candidates, with the held-out test evaluated exactly once per seed
on the validation winner.

Rows and their grids (single source of truth for the tuning protocol)
--------------------------------------------------------------------
    --row x --backbone bqn      BQNSearchSpace default grid: depth {1..5} x
                                activation {gelu, leaky_relu, elu} x dropout
                                {0, .1, .2, .3}  = 60 candidates
    --row x --backbone alter    collaborator's ALTER variant (nhead 1,
                                corr > 0.3, bias-free) x n_layers {1, 2} x
                                dropout {0.1, 0.3}                 = 4
    --row x --backbone brainnetcnn
                                Kawahara BrainNetCNN (models/brainnetcnn) x
                                dropout {0.1, 0.3, 0.5} x lr {1e-4, 3e-4, 1e-3} = 9
    --row x --backbone bnt      supplied GraphTransformer (mean readout, 4 heads,
                                FFN 1024), layers {1,2} x dropout {.1,.3}
                                x lr {1e-4,3e-4} = 8 candidates
    --row x --backbone gcn      GCN depth {1,2,3} x hidden {32,64,128} x dropout {.1,.3}
                                x lr {1e-4,1e-3} = 36 candidates
    --row templasso             C {0.005, 0.01, 0.03} (ABIDE) / {0.01, 0.03, 0.1}
                                (ADNI: C=0.005 keeps no edges on 96x12 rows)
                                x BiGRU hidden {32, 64, 128} x lr {1e-4, 3e-4} = 18
    --row tempx --backbone B    TempX = B applied per window -> BiGRU. Small
                                backbone grid (bqn: depth {1,3} x dropout {.1,.3},
                                elu; alter: harness variant, n_layers {1,2} x
                                dropout {.1,.3}; brainnetcnn: dropout {.3,.5})
                                x BiGRU hidden {32, 64, 128} x lr {1e-4, 3e-4}
                                = 24 / 24 / 12
                                bnt: layers {1,2} x dropout {.1,.3}, same BiGRU
                                and learning-rate axes = 24 candidates
                                gcn: depth {2,3} x hidden {32,64} x dropout {.1,.3}
                                x BiGRU hidden {32,64,128} x lr {1e-4,1e-3} = 48

Joint X + TempLasso and X + TempX are not tuned by this script. They are
available through unified_bn.training.abide_cli and the candidate-generation
API. The separate late_fusion_from_checkpoints.py experiment combines
predictions from independently selected branches; it is a different method.

Why this is not a single run_validation_search() call
------------------------------------------------------
The orchestrator trains a candidate list sequentially in one process. A
60-candidate x 5-seed grid needs to run ~40-wide on a node, so every
(candidate, seed) is trained as its own single-candidate search in its own
directory, and ``select`` then applies the orchestrator's rule -- minimum
validation loss, ties to the lowest candidate index -- across the finished
directories. The candidate list is generated deterministically from the
arguments, so every worker and the selector agree on index -> candidate.

Sub-commands
------------
    list      print the number of (candidate, seed) tasks
    train     --task-index i     train one task (resumable)
    select    per seed: pick the winner on validation, test it ONCE, write
              <grid>/seed<seed>/selected -> winner dir, and a CSV of rows
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product
from pathlib import Path

SEEDS = (42, 1042, 2042, 3042, 4042)
ROWS = ("x", "tempx", "templasso", "fusion_tempx", "fusion_templasso")
BACKBONES = ("bqn", "alter", "brainnetcnn", "bnt", "gcn")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("list", "train", "select"))
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--row", choices=ROWS, required=True)
    ap.add_argument("--backbone", choices=BACKBONES, default="bqn")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-prefix", required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--static-h5", type=Path, required=True)
    ap.add_argument("--temporal-h5", type=Path, required=True)
    ap.add_argument("--grid-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expected-n-rois", type=int, default=100)
    ap.add_argument("--select-metric", choices=("val_loss", "val_auc"), default="val_loss",
                    help="checkpoint (epoch) AND candidate selection metric; val_auc = macro OvR AUC for K>2")
    ap.add_argument("--select-mode", choices=("min", "max"), default=None,
                    help="default: min for val_loss, max for val_auc")
    ap.add_argument("--num-classes", type=int, default=2,
                    help="2 = binary ROC-AUC protocol; K>2 = macro one-vs-rest AUC (e.g. PPMI 4-class)")
    ap.add_argument("--tempx-grid", choices=("default", "surbhi", "surbhi_l1"), default="default",
                    help="tempx only. surbhi = the collaborator's temporal search space: BiGRU h {32,64} x layers {1,2} "
                         "x stacked dropout {.1,.3} x pooling {final, attention}, lr 1e-4; backbone axis = her BQN "
                         "{3,4} x elu x dropout {.2,.3}, or the X grid for the other backbones. Grid dir gets _sgrid. "
                         "surbhi_l1 = the same with 1-layer BiGRU only (her 2-layer candidates never won on ABIDE); dir _sgrid1.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--lasso-cs", type=float, nargs="+", default=None,
                    help="TempLasso C grid. Default: ABIDE 0.005/0.01/0.03; otherwise 0.01/0.03/0.1 "
                         "(on ADNI's 96x12 training rows C=0.005 keeps no edges).")
    ap.add_argument("--out", type=Path, default=None, help="select: CSV of selected rows")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo.expanduser().resolve()))

    from unified_bn.temporal.windowing import WindowPolicyConfig
    from unified_bn.training.config import (
        EarlyStoppingConfig, LRScheduleConfig, OptimizerConfig, TrainingConfig,
        ValidationSelectionRule,
    )
    # getattr: callers that build a bare Namespace (tests) keep the historical val_loss/min rule.
    select_metric = getattr(args, "select_metric", None) or "val_loss"
    select_mode = getattr(args, "select_mode", None) or ("min" if select_metric == "val_loss" else "max")
    selection_rule = ValidationSelectionRule(metric=select_metric, mode=select_mode)
    tempx_grid = getattr(args, "tempx_grid", None) or "default"
    from unified_bn.training.search_space import (
        ALTERSearchSpace, BQNSearchSpace, BiGRUSearchSpace, BrainNetCNNSearchSpace,
        BNTSearchSpace,
        GCNSearchSpace,
        LassoEdgeSearchSpace, candidate_fingerprint, generate_templasso_candidates,
        generate_temporal_candidates, generate_vanilla_alter_candidates,
        generate_fusion_candidates, generate_fusion_templasso_candidates,
        generate_vanilla_bqn_candidates, generate_vanilla_candidates,
    )
    from unified_bn.data import load_frozen_split
    from unified_bn.training.abide_runtime import (
        ABIDERuntimeConfig, ABIDERuntimeFactory, frozen_split_identity,
    )

    def training(lr: float) -> TrainingConfig:
        return TrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size, num_workers=0,
            optimizer=OptimizerConfig(lr=lr, weight_decay=1e-4),
            lr_schedule=LRScheduleConfig(name="cosine", min_lr=min(1e-5, lr)),
            checkpoint_selection=selection_rule,
            early_stopping=EarlyStoppingConfig(rule=selection_rule),
        )

    def candidates_for(seed: int, split_identity: str):
        if args.row == "x" and args.backbone == "bqn":
            return generate_vanilla_bqn_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                backbone_space=BQNSearchSpace(n_rois=args.expected_n_rois),
                training_configs=(training(1e-4),),
            )
        if args.row == "x" and args.backbone == "alter":
            return generate_vanilla_alter_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                backbone_space=ALTERSearchSpace(
                    n_layers=(1, 2), dropouts=(0.1, 0.3), nhead=1,
                    edge_thresholds=(0.3,), transformer_bias=False,
                    n_rois=args.expected_n_rois,
                ),
                training_configs=(training(1e-4),),
            )
        if args.row == "x" and args.backbone == "brainnetcnn":
            return generate_vanilla_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                backbone_space=BrainNetCNNSearchSpace(dropouts=(0.1, 0.3, 0.5),
                                                      n_rois=args.expected_n_rois),
                training_configs=(training(1e-4), training(3e-4), training(1e-3)),
            )
        if args.row == "x" and args.backbone == "bnt":
            return generate_vanilla_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                backbone_space=BNTSearchSpace(n_layers=(1, 2), dropouts=(0.1, 0.3),
                                             n_rois=args.expected_n_rois),
                training_configs=(training(1e-4), training(3e-4)),
            )
        if args.row == "x" and args.backbone == "gcn":
            # depth {1,2,3} x width {32,64,128} x dropout {.1,.3} x lr {1e-4,1e-3} = 36
            return generate_vanilla_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                backbone_space=GCNSearchSpace(n_layers=(1, 2, 3), hidden_dims=(32, 64, 128),
                                             dropouts=(0.1, 0.3), n_rois=args.expected_n_rois),
                training_configs=(training(1e-4), training(1e-3)),
            )
        if args.row == "tempx" and tempx_grid in ("surbhi", "surbhi_l1"):
            # The collaborator's ppmi-multiclass temporal grid (temporal_bqn_production.py), applied to our backbones.
            if args.backbone == "bqn":
                backbone_space = BQNSearchSpace(n_layers=(3, 4), activations=("elu",), dropouts=(0.2, 0.3),
                                                n_rois=args.expected_n_rois)
            elif args.backbone == "alter":
                backbone_space = ALTERSearchSpace(n_layers=(1, 2), dropouts=(0.1, 0.3), nhead=1,
                                                  edge_thresholds=(0.3,), transformer_bias=False,
                                                  n_rois=args.expected_n_rois)
            elif args.backbone == "bnt":
                backbone_space = BNTSearchSpace(n_layers=(1, 2), dropouts=(0.1, 0.3), n_rois=args.expected_n_rois)
            elif args.backbone == "gcn":
                backbone_space = GCNSearchSpace(n_layers=(1, 2, 3), hidden_dims=(32, 64, 128),
                                                dropouts=(0.1, 0.3), n_rois=args.expected_n_rois)
            else:
                backbone_space = BrainNetCNNSearchSpace(dropouts=(0.1, 0.3, 0.5), n_rois=args.expected_n_rois)
            return generate_temporal_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                window_policy=WindowPolicyConfig(policy="native"),
                temporal_space=(BiGRUSearchSpace(hidden_dims=(32, 64), num_layers=(1, 2),
                                                 poolings=("final", "attention"), stacked_dropouts=(0.1, 0.3),
                                                 attention_hidden_dims=(32,), attention_dropouts=(0.0,))
                                if tempx_grid == "surbhi" else
                                BiGRUSearchSpace(hidden_dims=(32, 64), num_layers=(1,),
                                                 poolings=("final", "attention"),
                                                 attention_hidden_dims=(32,), attention_dropouts=(0.0,))),
                backbone_space=backbone_space,
                training_configs=(training(1e-4),),
            )
        def tempx_backbone_space():
            if args.backbone == "bqn":
                return BQNSearchSpace(
                    n_layers=(1, 3), activations=("elu",), dropouts=(0.1, 0.3),
                    n_rois=args.expected_n_rois)
            if args.backbone == "alter":
                return ALTERSearchSpace(
                    n_layers=(1, 2), dropouts=(0.1, 0.3), nhead=1,
                    edge_thresholds=(0.3,), transformer_bias=False,
                    n_rois=args.expected_n_rois)
            if args.backbone == "bnt":
                return BNTSearchSpace(n_layers=(1, 2), dropouts=(0.1, 0.3),
                                      n_rois=args.expected_n_rois)
            if args.backbone == "gcn":
                # depth {2,3} x width {32,64} x dropout {.1,.3} x BiGRU {32,64,128} x lr {1e-4,1e-3} = 48
                return GCNSearchSpace(n_layers=(2, 3), hidden_dims=(32, 64),
                                      dropouts=(0.1, 0.3), n_rois=args.expected_n_rois)
            return BrainNetCNNSearchSpace(dropouts=(0.3, 0.5), n_rois=args.expected_n_rois)

        def lasso_cs():
            return tuple(args.lasso_cs) if args.lasso_cs else (
                (0.005, 0.01, 0.03) if args.dataset.upper() == "ABIDE" else (0.01, 0.03, 0.1)
            )

        narrow_bigru = BiGRUSearchSpace(hidden_dims=(32, 64, 128), num_layers=(1,), poolings=("final",))

        if args.row == "fusion_tempx":
            # X + TempX, concat joint training (the paper's "simple fusion"): the same backbone
            # architecture on both branches as two independent module instances, one classifier
            # on [h_static || h_temporal]. Same backbone sub-grid, BiGRU axis and lrs as the TempX row.
            return generate_fusion_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                window_policy=WindowPolicyConfig(policy="native"),
                temporal_space=narrow_bigru, backbone_space=tempx_backbone_space(),
                training_configs=(training(1e-4), training(1e-3 if args.backbone == "gcn" else 3e-4)),
            )
        if args.row == "fusion_templasso":
            # X + TempLasso, concat joint training: static branch = backbone (TempX sub-grid),
            # temporal branch = train-fitted lasso_edges encoder -> BiGRU; C x BiGRU x lr as TempLasso.
            return generate_fusion_templasso_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                window_policy=WindowPolicyConfig(policy="native"),
                temporal_space=narrow_bigru, backbone_space=tempx_backbone_space(),
                selector_space=LassoEdgeSearchSpace(Cs=lasso_cs(), n_rois=args.expected_n_rois),
                training_configs=(training(1e-4), training(3e-4)),
            )
        if args.row == "tempx":
            backbone_space = tempx_backbone_space()
            return generate_temporal_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                window_policy=WindowPolicyConfig(policy="native"),
                temporal_space=BiGRUSearchSpace(hidden_dims=(32, 64, 128), num_layers=(1,),
                                                poolings=("final",)),
                backbone_space=backbone_space,
                training_configs=(training(1e-4), training(1e-3 if args.backbone == "gcn" else 3e-4)),
            )
        if args.row == "templasso":
            cs = lasso_cs()
            return generate_templasso_candidates(
                seed=seed, split_identity=split_identity, num_classes=args.num_classes,
                window_policy=WindowPolicyConfig(policy="native"),
                temporal_space=BiGRUSearchSpace(hidden_dims=(32, 64, 128), num_layers=(1,),
                                                poolings=("final",)),
                selector_space=LassoEdgeSearchSpace(Cs=cs, n_rois=args.expected_n_rois),
                training_configs=(training(1e-4), training(3e-4)),
            )
        raise SystemExit(f"unsupported row/backbone {args.row}/{args.backbone}")

    def split_path(seed: int) -> Path:
        return args.split_dir.expanduser() / f"{args.split_prefix}_split_seed{seed}.npz"

    def runtime_factory(seed: int) -> ABIDERuntimeFactory:
        return ABIDERuntimeFactory(ABIDERuntimeConfig(
            split_path=split_path(seed), static_h5=args.static_h5.expanduser(),
            temporal_h5=args.temporal_h5.expanduser(), device=args.device,
            expected_n_rois=args.expected_n_rois, dataset_name=args.dataset,
            num_classes=args.num_classes,
        ))

    row_tag = "templasso" if args.row == "templasso" else f"{args.row}_{args.backbone}"
    if args.row == "tempx" and tempx_grid in ("surbhi", "surbhi_l1"):
        row_tag += "_sgrid" if tempx_grid == "surbhi" else "_sgrid1"
    # Non-default selection rules get their own grid directory (their fingerprints differ anyway).
    rule_tag = "" if (select_metric, select_mode) == ("val_loss", "min") else f"_{select_metric}_{select_mode}"
    grid_root = args.grid_dir.expanduser() / f"{args.dataset.lower()}_{row_tag}{rule_tag}"

    # Enumerate tasks deterministically: seed-major, candidate-minor.
    tasks = []
    per_seed_candidates = {}
    for seed in args.seeds:
        ident = frozen_split_identity(load_frozen_split(split_path(seed)))
        cands = candidates_for(seed, ident)
        per_seed_candidates[seed] = cands
        for index, cand in enumerate(cands):
            tasks.append((seed, index, cand))

    if args.command == "list":
        print(len(tasks))
        print(f"# {len(args.seeds)} seeds x {len(per_seed_candidates[args.seeds[0]])} candidates "
              f"({args.dataset} {row_tag}, {args.epochs} epochs) -> {grid_root}", file=sys.stderr)
        return 0

    def task_dir(seed: int, index: int, cand) -> Path:
        return grid_root / f"seed{seed}" / f"cand_{index:04d}_{candidate_fingerprint(cand)[:12]}"

    if args.command == "train":
        from unified_bn.training.orchestrator import run_validation_search
        if args.task_index is None:
            raise SystemExit("train needs --task-index")
        seed, index, cand = tasks[args.task_index]
        d = task_dir(seed, index, cand)
        d.mkdir(parents=True, exist_ok=True)
        selection = run_validation_search((cand,), search_dir=d, runtime_factory=runtime_factory(seed))
        print(f"task {args.task_index}: seed {seed} cand {index} -> val {selection.selection_metric}="
              f"{selection.selection_value:.5f}", flush=True)
        return 0

    # ---------------------------------------------------------------- select
    import torch
    from torch.utils.data import DataLoader
    from unified_bn.data import (
        FusionABIDEDataset, StaticABIDEDataset, TemporalABIDEDataset,
        collate_fusion, collate_static, collate_temporal,
    )
    from unified_bn.training.checkpoint import read_training_checkpoint
    from unified_bn.training.trainer import evaluate_final_test_and_record

    out_rows = []
    for seed in args.seeds:
        cands = per_seed_candidates[seed]
        scored = []
        for index, cand in enumerate(cands):
            d = task_dir(seed, index, cand)
            results = sorted(d.glob("candidate_*/result.json"))
            if not results:
                raise SystemExit(f"seed {seed}: candidate {index} has no result yet ({d})")
            payload = json.loads(results[0].read_text())
            if payload.get("status") not in ("validation_complete", "final_test_complete"):
                raise SystemExit(f"seed {seed}: candidate {index} incomplete: {payload.get('status')}")
            scored.append((float(payload["best_selection_value"]), index, cand, results[0]))
        # orchestrator rule: best selection value (min val_loss / max val_auc), ties -> lowest index
        if select_mode == "max":
            best_value, best_index, best_cand, best_result = max(scored, key=lambda t: (t[0], -t[1]))
        else:
            best_value, best_index, best_cand, best_result = min(scored, key=lambda t: (t[0], t[1]))
        winner_dir = best_result.parent

        link = grid_root / f"seed{seed}" / "selected"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(winner_dir.parent, target_is_directory=True)

        payload = json.loads(best_result.read_text())
        if payload.get("test_metrics") is None:
            rt = runtime_factory(seed)(best_cand, 0, winner_dir)  # rebuild (re-fits selector on TRAIN)
            model = rt.model
            state = read_training_checkpoint(winner_dir / "best.pt", expected_run_config=best_cand).model_state
            missing, unexpected = model.load_state_dict(state, strict=True)
            assert not missing and not unexpected
            model.to(args.device).eval()
            split = load_frozen_split(split_path(seed))
            if best_cand.identity.mode == "vanilla":
                ds = StaticABIDEDataset(args.static_h5.expanduser(), subject_ids=split.test_ids,
                                        expected_n_rois=args.expected_n_rois, num_classes=args.num_classes); collate = collate_static
            elif best_cand.identity.mode == "fusion":
                ds = FusionABIDEDataset(args.static_h5.expanduser(), args.temporal_h5.expanduser(),
                                        subject_ids=split.test_ids, expected_n_rois=args.expected_n_rois,
                                        num_classes=args.num_classes); collate = collate_fusion
            else:
                ds = TemporalABIDEDataset(args.temporal_h5.expanduser(), subject_ids=split.test_ids,
                                          expected_n_rois=args.expected_n_rois, num_classes=args.num_classes); collate = collate_temporal
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
            evaluate_final_test_and_record(model, loader, result_path=best_result, device=args.device)
            payload = json.loads(best_result.read_text())
        tm = payload["test_metrics"]; vm = payload["best_validation_metrics"]
        ident = best_cand.identity
        cfg = dict(ident.backbone_config)
        summary = {k: cfg[k] for k in cfg if k in ("n_layers", "activation", "dropout", "C", "nhead", "edge_threshold")}
        if args.backbone == "bnt" and args.row != "templasso":
            summary.update(readout=cfg["readout"], ffn_dim=cfg["ffn_dim"])
        if args.backbone == "gcn" and args.row != "templasso":
            summary.update(gcn_hidden_dim=cfg["hidden_dim"], adjacency=cfg["adjacency"],
                           node_features=cfg["node_features"], readout=cfg["readout"])
        if ident.temporal_config: summary["hidden_dim"] = ident.temporal_config["hidden_dim"]
        if ident.temporal_backbone_config and "C" in ident.temporal_backbone_config:
            summary["C"] = ident.temporal_backbone_config["C"]
        summary["lr"] = best_cand.training.optimizer.lr
        out_rows.append({
            "dataset": args.dataset,
            "row": {"x": "X", "tempx": "TempX", "templasso": "TempLasso",
                    "fusion_tempx": "X+TempX (concat)", "fusion_templasso": "X+TempLasso (concat)"}[args.row],
            "backbone": "" if args.row == "templasso" else args.backbone, "seed": seed,
            "n_candidates": len(cands), "selected_index": best_index, "selected_config": json.dumps(summary),
            "selection_rule": f"{select_metric}/{select_mode}",
            "val_loss": vm["loss"], "val_auc": vm["roc_auc"], "best_epoch": payload["best_epoch"],
            "test_auc": tm["roc_auc"], "test_accuracy": tm["accuracy"],
            "test_sensitivity": tm["sensitivity"], "test_specificity": tm["specificity"],
            "winner_dir": str(winner_dir),
        })
        print(f"seed {seed}: winner cand {best_index} {summary} val_loss={best_value:.4f} "
              f"test_auc={tm['roc_auc']:.4f}", flush=True)

    out = args.out or (grid_root / "selected_rows.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    aucs = [r["test_auc"] for r in out_rows]
    import statistics as st
    print(f"\n{args.dataset} {row_tag}: test AUC {st.fmean(aucs)*100:.2f} ± {st.pstdev(aucs)*100:.2f} "
          f"over {len(aucs)} seeds  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
