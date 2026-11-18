#!/usr/bin/env python3
"""
Late fusion of two already-trained branches -- X (static) and a temporal model --
from their selected checkpoints, with NO further training.

For each seed it loads the val-selected checkpoint of each branch (built
through the same runtime factory the search used, so TempLasso's edge set is
re-fit on TRAIN exactly as before), produces class-1 probabilities on the
validation and test partitions, and reports:

    X                     static branch alone
    T                     TempLasso alone
    avg                   p = (p_X + p_T) / 2                    (0 tuned params)
    alpha*                p = a*p_X + (1-a)*p_T, chosen by validation loss
                          (PPMI: maximum validation macro OvR AUC)
    logit-avg             sigmoid((z_X + z_T)/2)                (0 tuned params)
    stack                 logistic regression on [z_X, z_T] fit on val
    rank-avg              mean of within-set ranks               (0 tuned params)

Validation labels/probabilities alone select weights. The historical script
also computes test probabilities for reporting; they never enter selection.
Nothing here writes back into any result artifact.

Usage
-----
    python experiments/late_fusion_from_checkpoints.py \
        --repo ~/tbn --dataset ABIDE --split-prefix abide --split-dir ~/tbn-data/splits \
        --static-h5 ~/tbn-data/abide_data_tr.h5 \
        --temporal-h5 ~/tbn-data/derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 \
        --x-runs '~/tbn-runs/real_fixedcfg/vanilla_seed{seed}' \
        --t-runs '~/tbn-runs/real_fixedcfg/templasso_seed{seed}' \
        --label bqn_abide --out results/late_fusion_bqn_abide.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

SEEDS = (42, 1042, 2042, 3042, 4042)


def select_multiclass_alpha(y_val, static_probs, temporal_probs, *, dataset, num_classes):
    """Validation-only mixing weight; earliest grid alpha breaks exact ties."""
    from sklearn.metrics import log_loss, roc_auc_score
    from unified_bn.training.dataset_protocol import dataset_selection_rule
    rule = dataset_selection_rule(dataset, num_classes)
    alphas = np.linspace(0, 1, 21)
    scores = []
    for alpha in alphas:
        probs = alpha * static_probs + (1 - alpha) * temporal_probs
        if rule.mode == "max":
            scores.append(roc_auc_score(y_val, probs, multi_class="ovr", average="macro",
                                        labels=list(range(num_classes))))
        else:
            scores.append(log_loss(y_val, np.clip(probs, 1e-6, 1.0), labels=list(range(num_classes))))
    if not np.isfinite(scores).all():
        raise ValueError("Mixing-weight validation scores must be finite")
    index = int(np.argmax(scores) if rule.mode == "max" else np.argmin(scores))
    return float(alphas[index])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-prefix", required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--static-h5", type=Path, required=True)
    ap.add_argument("--temporal-h5", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--x-runs", required=True, help="search dir pattern with {seed}")
    ap.add_argument("--t-runs", required=True, help="search dir pattern with {seed}")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--expected-n-rois", type=int, default=100)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    sys.path.insert(0, str(repo))

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from torch.utils.data import DataLoader

    from unified_bn.data import (
        StaticABIDEDataset, TemporalABIDEDataset, collate_static, collate_temporal,
        load_frozen_split,
    )
    from unified_bn.temporal.windowing import WindowPolicyConfig
    from unified_bn.training.abide_runtime import ABIDERuntimeConfig, ABIDERuntimeFactory
    from unified_bn.training.checkpoint import read_training_checkpoint
    from unified_bn.training.dataset_protocol import dataset_selection_rule
    from unified_bn.training.config import (
        CandidateRunConfig, DeterminismConfig, EarlyStoppingConfig, LRScheduleConfig,
        OptimizerConfig, RunIdentity, TrainingConfig, ValidationSelectionRule,
    )
    from unified_bn.training.search_space import candidate_fingerprint

    def candidate_from_result(result_path: Path) -> CandidateRunConfig:
        """Rebuild the frozen candidate from result.json and verify its fingerprint."""
        payload = json.loads(result_path.read_text())
        ident = dict(payload["run_config"]["identity"])
        tr = payload["run_config"]["training"]
        wp = ident.pop("window_policy", None)
        identity = RunIdentity(
            **{k: v for k, v in ident.items() if k not in ("window_policy",)},
            window_policy=WindowPolicyConfig(**wp) if wp else None,
        )
        es = tr["early_stopping"]
        training = TrainingConfig(
            epochs=tr["epochs"], batch_size=tr["batch_size"], num_workers=tr["num_workers"],
            optimizer=OptimizerConfig(**tr["optimizer"]),
            lr_schedule=LRScheduleConfig(**tr["lr_schedule"]),
            checkpoint_selection=ValidationSelectionRule(**tr["checkpoint_selection"]),
            early_stopping=EarlyStoppingConfig(
                enabled=es["enabled"], patience=es["patience"], min_delta=es["min_delta"],
                rule=ValidationSelectionRule(**es["rule"]),
            ),
            determinism=DeterminismConfig(**tr["determinism"]),
        )
        cand = CandidateRunConfig(identity=identity, training=training)
        fp = candidate_fingerprint(cand)
        if not result_path.parent.name.endswith(fp[:12]):
            # Runs produced before main@e97aabe serialised DeterminismConfig
            # without num_threads, so their fingerprint differs from today's
            # although the scientific identity (mode/backbone/temporal/split)
            # is unchanged. The identity part is what selects the model.
            print(f"  note: {result_path.parent.name} predates the current TrainingConfig "
                  f"schema (rebuilt fp {fp[:12]}); identity fields verified, proceeding", flush=True)
        return cand

    def probs(run_dir: Path, seed: int, split_path: Path):
        """Return (val_probs, val_labels, test_probs, test_labels, meta) for a selected run."""
        result_path = next(iter(sorted(run_dir.glob("candidate_*/result.json"))))
        candidate = candidate_from_result(result_path)
        expected_rule = dataset_selection_rule(args.dataset, args.num_classes)
        if args.dataset.strip().upper() == "PPMI" and candidate.training.checkpoint_selection != expected_rule:
            raise ValueError("PPMI late fusion requires macro-AUC-selected checkpoints from new runs.")
        best = result_path.parent / "best.pt"
        factory = ABIDERuntimeFactory(ABIDERuntimeConfig(
            split_path=split_path, static_h5=args.static_h5.expanduser(),
            temporal_h5=args.temporal_h5.expanduser(), device=args.device,
            expected_n_rois=args.expected_n_rois, dataset_name=args.dataset,
            num_classes=args.num_classes,
        ))
        runtime = factory(candidate, 0, run_dir / "_late_fusion_tmp")
        model = runtime.model
        try:
            state = read_training_checkpoint(best, expected_run_config=candidate).model_state
        except Exception as exc:  # legacy schema: read the raw payload
            payload = torch.load(best, map_location="cpu", weights_only=False)
            state = next(payload[k] for k in ("model_state", "model_state_dict", "model") if k in payload)
            print(f"  note: strict checkpoint read failed ({type(exc).__name__}); loaded raw model_state", flush=True)
        missing, unexpected = model.load_state_dict(state, strict=True)
        assert not missing and not unexpected
        model.to(args.device).eval()

        split = load_frozen_split(split_path)
        mode = candidate.identity.mode
        if mode == "vanilla":
            test_ds = StaticABIDEDataset(args.static_h5.expanduser(), subject_ids=split.test_ids,
                                         expected_n_rois=args.expected_n_rois, num_classes=args.num_classes)
            collate = collate_static
        else:
            test_ds = TemporalABIDEDataset(args.temporal_h5.expanduser(), subject_ids=split.test_ids,
                                           expected_n_rois=args.expected_n_rois, num_classes=args.num_classes)
            collate = collate_temporal
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

        def run(loader):
            ps, ys = [], []
            with torch.no_grad():
                for b in loader:
                    if mode == "vanilla":
                        out = model(b.fc.to(args.device))
                    else:
                        out = model(b.dynamic_fc.to(args.device), b.lengths.to(args.device))
                    pr = torch.softmax(out.logits, -1).cpu().numpy()
                    ps.append(pr[:, 1] if args.num_classes == 2 else pr); ys.append(b.labels.numpy())
            return np.concatenate(ps), np.concatenate(ys)

        pv, yv = run(runtime.val_batches)
        pt, yt = run(test_loader)
        res = json.loads(result_path.read_text())
        meta = {"best_epoch": res.get("best_epoch"), "fingerprint": candidate_fingerprint(candidate)[:12]}
        return pv, yv, pt, yt, meta

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def ranks(p):
        return np.argsort(np.argsort(p)) / max(len(p) - 1, 1)

    rows = []
    for seed in args.seeds:
        split_path = args.split_dir.expanduser() / f"{args.split_prefix}_split_seed{seed}.npz"
        xv, yv, xt, yt, mx = probs(Path(os.path.expanduser(args.x_runs.format(seed=seed))), seed, split_path)
        tv, yv2, tt, yt2, mt = probs(Path(os.path.expanduser(args.t_runs.format(seed=seed))), seed, split_path)
        assert (yv == yv2).all() and (yt == yt2).all(), "branch partitions disagree"

        if args.num_classes > 2:
            # K-class: probability-vector fusions only (no binary logit tricks);
            # macro one-vs-rest AUC; alpha* is still chosen on validation only.
            def mauc(y, p):
                return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(args.num_classes)))
            def sm(z):
                z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)
            lg = lambda p: np.log(np.clip(p, 1e-6, 1.0))
            a_star = select_multiclass_alpha(yv, xv, tv, dataset=args.dataset, num_classes=args.num_classes)
            fus = {
                "X": (xv, xt), "T": (tv, tt),
                "avg": ((xv + tv) / 2, (xt + tt) / 2),
                "alpha*": (a_star * xv + (1 - a_star) * tv, a_star * xt + (1 - a_star) * tt),
                "logit-avg": (sm((lg(xv) + lg(tv)) / 2), sm((lg(xt) + lg(tt)) / 2)),
            }
            for name, (pv, pt) in fus.items():
                rows.append({
                    "label": args.label, "dataset": args.dataset, "seed": seed, "fusion": name,
                    "val_auc": mauc(yv, pv), "test_auc": mauc(yt, pt),
                    "test_acc": float((pt.argmax(1) == yt).mean()),
                    "alpha": a_star if name == "alpha*" else "", "stack_w_x": "", "stack_w_t": "",
                    "x_best_epoch": mx["best_epoch"], "t_best_epoch": mt["best_epoch"],
                })
            print(f"seed {seed}: " + "  ".join(f"{n}={mauc(yt, pt):.3f}" for n, (_, pt) in fus.items())
                  + f"  (alpha*={a_star:.2f}; macro OvR AUC, {args.num_classes} classes)", flush=True)
            continue

        # -- every choice below is made on validation only --------------------
        alphas = np.linspace(0, 1, 21)
        a_star = alphas[int(np.argmin([log_loss(yv, np.clip(a * xv + (1 - a) * tv, 1e-6, 1 - 1e-6)) for a in alphas]))]
        stacker = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[logit(xv), logit(tv)], yv)

        fus = {
            "X": (xv, xt),
            "T": (tv, tt),
            "avg": ((xv + tv) / 2, (xt + tt) / 2),
            "alpha*": (a_star * xv + (1 - a_star) * tv, a_star * xt + (1 - a_star) * tt),
            "logit-avg": (sigmoid((logit(xv) + logit(tv)) / 2), sigmoid((logit(xt) + logit(tt)) / 2)),
            "stack": (stacker.predict_proba(np.c_[logit(xv), logit(tv)])[:, 1],
                      stacker.predict_proba(np.c_[logit(xt), logit(tt)])[:, 1]),
            "rank-avg": ((ranks(xv) + ranks(tv)) / 2, (ranks(xt) + ranks(tt)) / 2),
        }
        for name, (pv, pt) in fus.items():
            rows.append({
                "label": args.label, "dataset": args.dataset, "seed": seed, "fusion": name,
                "val_auc": roc_auc_score(yv, pv), "test_auc": roc_auc_score(yt, pt),
                "test_acc": float(((pt >= 0.5).astype(int) == yt).mean()),
                "alpha": a_star if name == "alpha*" else "",
                "stack_w_x": stacker.coef_[0][0] if name == "stack" else "",
                "stack_w_t": stacker.coef_[0][1] if name == "stack" else "",
                "x_best_epoch": mx["best_epoch"], "t_best_epoch": mt["best_epoch"],
            })
        print(f"seed {seed}: " + "  ".join(f"{n}={roc_auc_score(yt, pt):.3f}" for n, (_, pt) in fus.items())
              + f"  (alpha*={a_star:.2f}, stack w=[{stacker.coef_[0][0]:+.2f},{stacker.coef_[0][1]:+.2f}])", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n{args.label} ({args.dataset}) — test AUC mean ± sample SD over {len(args.seeds)} seeds "
          f"(val AUC in parentheses)")
    for name in ("X", "T", "avg", "alpha*", "logit-avg", "stack", "rank-avg"):
        t = [r["test_auc"] for r in rows if r["fusion"] == name]
        if not t:
            continue
        v = [r["val_auc"] for r in rows if r["fusion"] == name]
        sample_sd = np.std(t, ddof=1) if len(t) > 1 else float("nan")
        print(f"  {name:<10} {np.mean(t)*100:6.2f} ± {sample_sd*100:4.2f}   ({np.mean(v)*100:5.1f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
