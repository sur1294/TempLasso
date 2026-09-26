#!/usr/bin/env python3
"""
Train ONE temporal-only BQN + BiGRU configuration on ONE frozen ABIDE
Schaefer100 split.

Model
-----
Temporal FC windows
    -> original BQN backbone on each valid window
    -> BiGRU + masked attention pooling
    -> temporal subject embedding
    -> lightweight classifier

This script intentionally contains NO static whole-scan branch and NO fusion.

Research protocol
-----------------
- Uses the existing temporal HDF5 only.
- Uses an existing frozen subject-ID split.
- Never creates a new split.
- Trains on train subjects only.
- Uses validation loss for checkpoint selection.
- Does NOT evaluate test unless --evaluate-test is explicitly provided.
- Keeps native K; batching/padding is handled by temporal_dataset.py.
- No ComBat/scaling/feature fitting is performed here.
- Cross-entropy is unweighted by default because ABIDE imbalance is mild.
- Optional gradient clipping is supported for recurrent stability.
- Optional label smoothing is exposed but defaults to 0.
- Optional static-BQN checkpoint initialization is supported as an explicit
  ablation; default is training from scratch.

Recommended workflow
--------------------
1. Smoke test one configuration for a few epochs WITHOUT --evaluate-test.
2. Hyperparameter tuning: all candidate runs WITHOUT --evaluate-test.
3. Select each seed's best configuration by minimum validation loss.
4. Evaluate only the selected checkpoint on the frozen test set.

Run from Brainnetwork root, e.g.

python models/bqn/bqn_plus/train_temporal_bqn_bigru.py \
    --temporal-h5 derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 \
    --split splits/abide_split_seed42.npz \
    --seed 42 \
    --bqn-layers 3 \
    --bqn-dropout 0.3 \
    --bqn-activation elu \
    --bigru-hidden-dim 32 \
    --bigru-num-layers 1 \
    --bigru-dropout 0.1 \
    --attention-hidden-dim 32 \
    --classifier-hidden-dim 64 \
    --classifier-dropout 0.1 \
    --epochs 200 \
    --out-dir results/temporal_bqn_bigru
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BQN_DIR = _THIS_DIR.parent

import sys

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_BQN_DIR) not in sys.path:
    sys.path.insert(0, str(_BQN_DIR))

from temporal_dataset import build_temporal_loaders, load_frozen_split
from temporal_bqn_bigru import (
    TemporalBQNBiGRUConfig,
    TemporalBQNBiGRUClassifier,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum",
        label_smoothing=label_smoothing,
    )

    total_loss = 0.0
    n_total = 0

    logits_all = []
    labels_all = []

    for batch in loader:
        dynamic_fc = batch["dynamic_fc"].to(
            device,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )
        lengths = batch["lengths"].to(
            device,
            non_blocking=True,
        )
        valid_mask = batch["valid_mask"].to(
            device,
            non_blocking=True,
        )

        logits = model(
            dynamic_fc=dynamic_fc,
            lengths=lengths,
            valid_mask=valid_mask,
        )

        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Non-finite logits encountered during evaluation."
            )

        total_loss += float(
            criterion(logits, labels).item()
        )
        n_total += int(labels.numel())

        logits_all.append(
            logits.detach().cpu()
        )
        labels_all.append(
            labels.detach().cpu()
        )

    if n_total == 0:
        raise RuntimeError("Evaluation loader is empty.")

    logits = torch.cat(
        logits_all,
        dim=0,
    )
    labels = torch.cat(
        labels_all,
        dim=0,
    ).numpy()

    probs = torch.softmax(
        logits,
        dim=1,
    )[:, 1].numpy()

    preds = torch.argmax(
        logits,
        dim=1,
    ).numpy()

    return {
        "loss": total_loss / n_total,
        "auc": float(
            roc_auc_score(
                labels,
                probs,
            )
        ),
        "acc": float(
            accuracy_score(
                labels,
                preds,
            )
        ),
        "sens": float(
            recall_score(
                labels,
                preds,
                pos_label=1,
                zero_division=0,
            )
        ),
        "spec": float(
            recall_score(
                labels,
                preds,
                pos_label=0,
                zero_division=0,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def set_epoch_lr(
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    epochs: int,
    base_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    if schedule == "cosine":
        if epochs <= 1:
            lr = min_lr
        else:
            progress = epoch_index / (epochs - 1)
            lr = (
                min_lr
                + 0.5
                * (base_lr - min_lr)
                * (1.0 + math.cos(math.pi * progress))
            )
    elif schedule == "constant":
        lr = base_lr
    else:
        raise ValueError(
            f"Unknown LR schedule: {schedule}"
        )

    for group in optimizer.param_groups:
        group["lr"] = lr

    return float(lr)


# ---------------------------------------------------------------------------
# Optional BQN initialization
# ---------------------------------------------------------------------------

def load_bqn_initialization(
    model: TemporalBQNBiGRUClassifier,
    checkpoint_path: str,
) -> None:
    """
    Initialize only the BQN backbone from an established static BQN checkpoint.

    This is an OPTIONAL experimental condition, not the default temporal model.

    Expected static checkpoint format:
        {
            "model_state_dict": ...,
            ...
        }

    The full StaticBQN state is loaded into model.bqn.
    Its original classifier weights may be present in the checkpoint, but the
    temporal model never calls that classifier; it uses return_embedding=True.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint.get(
        "model_state_dict",
        None,
    )

    if state is None:
        raise RuntimeError(
            f"{checkpoint_path} has no model_state_dict."
        )

    model.bqn.load_state_dict(
        state,
        strict=True,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def global_grad_norm(
    parameters,
) -> float:
    grads = [
        p.grad.detach()
        for p in parameters
        if p.grad is not None
    ]

    if not grads:
        return 0.0

    device = grads[0].device

    norms = torch.stack(
        [
            torch.norm(
                g,
                p=2,
            ).to(device)
            for g in grads
        ]
    )

    return float(
        torch.norm(
            norms,
            p=2,
        ).item()
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    label_smoothing: float,
):
    model.train()

    total_loss = 0.0
    n_total = 0

    grad_norm_sum = 0.0
    n_steps = 0

    for batch in loader:
        dynamic_fc = batch["dynamic_fc"].to(
            device,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )
        lengths = batch["lengths"].to(
            device,
            non_blocking=True,
        )
        valid_mask = batch["valid_mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            dynamic_fc=dynamic_fc,
            lengths=lengths,
            valid_mask=valid_mask,
        )

        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Non-finite logits during training."
            )

        loss = nn.functional.cross_entropy(
            logits,
            labels,
            reduction="mean",
            label_smoothing=label_smoothing,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss."
            )

        loss.backward()

        raw_grad_norm = global_grad_norm(
            model.parameters()
        )

        if not np.isfinite(raw_grad_norm):
            raise FloatingPointError(
                "Non-finite gradient norm."
            )

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        batch_n = int(
            labels.numel()
        )

        total_loss += (
            float(loss.item())
            * batch_n
        )
        n_total += batch_n

        grad_norm_sum += raw_grad_norm
        n_steps += 1

    if n_total == 0:
        raise RuntimeError(
            "Training loader is empty."
        )

    return {
        "loss": total_loss / n_total,
        "mean_raw_grad_norm": (
            grad_norm_sum / max(n_steps, 1)
        ),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_run_dir(
    args,
) -> Path:
    layernorm_tag = (
        "ln1"
        if args.temporal_layer_norm
        else "ln0"
    )

    name = (
        f"seed{args.seed}"
        f"_bqnL{args.bqn_layers}"
        f"_{args.bqn_activation}"
        f"_bqndrop{args.bqn_dropout:g}"
        f"_gruH{args.bigru_hidden_dim}"
        f"_gruL{args.bigru_num_layers}"
        f"_grudrop{args.bigru_dropout:g}"
        f"_att{args.attention_hidden_dim}"
        f"_clfH{args.classifier_hidden_dim}"
        f"_clfdrop{args.classifier_dropout:g}"
        f"_{layernorm_tag}"
    )

    run_dir = Path(
        args.out_dir
    ) / name

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir


def save_history(
    history,
    path: Path,
) -> None:
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "mean_raw_grad_norm",
        "val_loss",
        "val_auc",
        "val_acc",
        "val_sens",
        "val_spec",
        "is_best",
    ]

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def train_run(args):
    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.lr <= 0:
        raise ValueError(
            "--lr must be positive."
        )

    if args.lr_min <= 0:
        raise ValueError(
            "--lr-min must be positive."
        )

    if args.lr_min > args.lr:
        raise ValueError(
            "--lr-min cannot exceed --lr."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay must be nonnegative."
        )

    if args.grad_clip < 0:
        raise ValueError(
            "--grad-clip must be >= 0."
        )

    if not 0 <= args.label_smoothing < 1:
        raise ValueError(
            "--label-smoothing must satisfy 0 <= x < 1."
        )

    # ------------------------------------------------------------------
    # Verify split seed
    # ------------------------------------------------------------------
    split = load_frozen_split(
        args.split
    )

    if "seed" not in split:
        raise RuntimeError(
            "Frozen split file has no stored seed."
        )

    split_seed = int(
        np.asarray(
            split["seed"]
        ).item()
    )

    if split_seed != args.seed:
        raise ValueError(
            f"--seed={args.seed}, but split stores seed={split_seed}."
        )

    fix_seed(
        args.seed
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = build_temporal_loaders(
        temporal_h5=args.temporal_h5,
        split_path=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = TemporalBQNBiGRUConfig(
        bqn_layers=args.bqn_layers,
        bqn_dropout=args.bqn_dropout,
        bqn_activation=args.bqn_activation,
        cluster_num=args.cluster_num,
        bigru_hidden_dim=args.bigru_hidden_dim,
        bigru_num_layers=args.bigru_num_layers,
        bigru_dropout=args.bigru_dropout,
        attention_hidden_dim=args.attention_hidden_dim,
        embedding_dim=32,
        temporal_layer_norm=args.temporal_layer_norm,
        classifier_hidden_dim=args.classifier_hidden_dim,
        classifier_dropout=args.classifier_dropout,
    )

    model = TemporalBQNBiGRUClassifier(
        config
    )

    if args.bqn_init_checkpoint is not None:
        load_bqn_initialization(
            model=model,
            checkpoint_path=args.bqn_init_checkpoint,
        )

    device = torch.device(
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    model = model.to(
        device
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = build_run_dir(
        args
    )

    best_path = (
        run_dir / "best_model.pt"
    )
    last_path = (
        run_dir / "last_model.pt"
    )
    history_path = (
        run_dir / "history.csv"
    )
    config_path = (
        run_dir / "run_config.json"
    )
    result_path = (
        run_dir / "result.json"
    )

    metadata = {
        "temporal_h5": str(
            Path(args.temporal_h5)
        ),
        "split_file": str(
            Path(args.split)
        ),
        "split_seed": split_seed,
        "model_seed": args.seed,
        "model_config": asdict(config),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "lr": args.lr,
            "lr_min": args.lr_min,
            "lr_schedule": args.lr_schedule,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "label_smoothing": args.label_smoothing,
            "checkpoint_rule": (
                "minimum validation cross-entropy loss"
            ),
            "early_stopping": False,
            "class_weights": False,
            "test_evaluated": bool(
                args.evaluate_test
            ),
            "bqn_initialization": (
                args.bqn_init_checkpoint
                if args.bqn_init_checkpoint
                else "random initialization"
            ),
        },
        "device": str(device),
        "parameter_summary": (
            model.parameter_summary()
        ),
    }

    with config_path.open(
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("=" * 82)
    print("TEMPORAL BQN + BiGRU TRAINING")
    print("=" * 82)
    print(f"device:             {device}")
    print(f"seed:               {args.seed}")
    print(f"split:              {args.split}")
    print(
        f"BQN:                "
        f"L={args.bqn_layers}, "
        f"act={args.bqn_activation}, "
        f"drop={args.bqn_dropout}"
    )
    print(
        f"BiGRU:              "
        f"H={args.bigru_hidden_dim}, "
        f"L={args.bigru_num_layers}, "
        f"drop={args.bigru_dropout}"
    )
    print(
        f"attention hidden:   "
        f"{args.attention_hidden_dim}"
    )
    print(
        f"classifier:         "
        f"H={args.classifier_hidden_dim}, "
        f"drop={args.classifier_dropout}, "
        f"LayerNorm={args.temporal_layer_norm}"
    )
    print(f"epochs:             {args.epochs}")
    print(f"batch size:         {args.batch_size}")
    print(
        f"LR:                 "
        f"{args.lr:g} -> {args.lr_min:g} "
        f"({args.lr_schedule})"
    )
    print(
        f"weight decay:       "
        f"{args.weight_decay:g}"
    )
    print(
        f"gradient clipping:  "
        f"{args.grad_clip:g}"
    )
    print(
        f"label smoothing:    "
        f"{args.label_smoothing:g}"
    )
    print(
        f"BQN init:           "
        f"{args.bqn_init_checkpoint or 'random'}"
    )
    print(
        f"parameters:         "
        f"{model.parameter_summary()}"
    )
    print(
        f"evaluate test now:  "
        f"{args.evaluate_test}"
    )
    print("=" * 82)

    best_val_loss = float(
        "inf"
    )
    best_epoch: Optional[int] = None
    best_val_metrics: Optional[
        Dict[str, float]
    ] = None

    history = []
    start_epoch = 0

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(
                f"--resume requested but {last_path} does not exist."
            )

        state = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            state["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )

        start_epoch = int(
            state["epoch"]
        )
        best_val_loss = float(
            state["best_val_loss"]
        )
        best_epoch = state.get(
            "best_epoch"
        )
        best_val_metrics = state.get(
            "best_val_metrics"
        )

        if history_path.exists():
            with history_path.open(
                "r"
            ) as f:
                history = list(
                    csv.DictReader(f)
                )

        print(
            f"[RESUME] continuing from epoch {start_epoch + 1}"
        )

    start_time = time.time()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch_idx in range(
        start_epoch,
        args.epochs,
    ):
        lr = set_epoch_lr(
            optimizer=optimizer,
            epoch_index=epoch_idx,
            epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.lr_min,
            schedule=args.lr_schedule,
        )

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            label_smoothing=args.label_smoothing,
        )

        is_best = (
            val_metrics["loss"]
            < best_val_loss
        )

        if is_best:
            best_val_loss = float(
                val_metrics["loss"]
            )
            best_epoch = (
                epoch_idx + 1
            )
            best_val_metrics = dict(
                val_metrics
            )

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "model_config": (
                        asdict(config)
                    ),
                    "seed": args.seed,
                    "split_file": str(
                        args.split
                    ),
                    "best_epoch": (
                        best_epoch
                    ),
                    "best_val_metrics": (
                        best_val_metrics
                    ),
                },
                best_path,
            )

        row = {
            "epoch": epoch_idx + 1,
            "lr": lr,
            "train_loss": (
                train_stats["loss"]
            ),
            "mean_raw_grad_norm": (
                train_stats[
                    "mean_raw_grad_norm"
                ]
            ),
            "val_loss": (
                val_metrics["loss"]
            ),
            "val_auc": (
                val_metrics["auc"]
            ),
            "val_acc": (
                val_metrics["acc"]
            ),
            "val_sens": (
                val_metrics["sens"]
            ),
            "val_spec": (
                val_metrics["spec"]
            ),
            "is_best": int(
                is_best
            ),
        }

        history.append(
            row
        )

        save_history(
            history,
            history_path,
        )

        # Last checkpoint is for fault-tolerant resume only.
        torch.save(
            {
                "epoch": epoch_idx + 1,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "best_val_loss": (
                    best_val_loss
                ),
                "best_epoch": (
                    best_epoch
                ),
                "best_val_metrics": (
                    best_val_metrics
                ),
                "model_config": (
                    asdict(config)
                ),
            },
            last_path,
        )

        print(
            f"epoch {epoch_idx + 1:03d}/{args.epochs} "
            f"lr={lr:.3e} "
            f"train_loss={train_stats['loss']:.5f} "
            f"grad={train_stats['mean_raw_grad_norm']:.3f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
            + (
                "  *BEST*"
                if is_best
                else ""
            )
        )

    elapsed = (
        time.time()
        - start_time
    )

    if (
        best_epoch is None
        or best_val_metrics is None
        or not best_path.exists()
    ):
        raise RuntimeError(
            "No best validation checkpoint was produced."
        )

    # ------------------------------------------------------------------
    # Restore best checkpoint and verify validation metrics
    # ------------------------------------------------------------------
    best_state = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_state[
            "model_state_dict"
        ],
        strict=True,
    )

    restored_val = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        label_smoothing=args.label_smoothing,
    )

    if not np.isclose(
        restored_val["loss"],
        float(
            best_val_metrics["loss"]
        ),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Restored best checkpoint does not reproduce "
            "the saved validation loss."
        )

    # ------------------------------------------------------------------
    # Optional final test
    # ------------------------------------------------------------------
    test_metrics = None

    if args.evaluate_test:
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            label_smoothing=args.label_smoothing,
        )

    result = {
        "seed": args.seed,
        "split_file": str(
            args.split
        ),
        "best_epoch": (
            best_epoch
        ),
        "best_val": (
            restored_val
        ),
        "test": (
            test_metrics
        ),
        "elapsed_seconds": (
            elapsed
        ),
        "model_config": (
            asdict(config)
        ),
        "training": (
            metadata["training"]
        ),
        "parameter_summary": (
            model.parameter_summary()
        ),
        "checkpoint": str(
            best_path
        ),
        "history": str(
            history_path
        ),
    }

    with result_path.open(
        "w"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 82)
    print("RUN COMPLETE")
    print("=" * 82)
    print(
        f"best epoch: {best_epoch}"
    )
    print(
        "best val: "
        f"loss={restored_val['loss']:.5f} "
        f"auc={restored_val['auc']:.4f} "
        f"acc={restored_val['acc']:.4f} "
        f"sens={restored_val['sens']:.4f} "
        f"spec={restored_val['spec']:.4f}"
    )

    if test_metrics is None:
        print(
            "test: NOT EVALUATED"
        )
        print(
            "Correct mode for hyperparameter tuning."
        )
    else:
        print(
            "test: "
            f"loss={test_metrics['loss']:.5f} "
            f"auc={test_metrics['auc']:.4f} "
            f"acc={test_metrics['acc']:.4f} "
            f"sens={test_metrics['sens']:.4f} "
            f"spec={test_metrics['spec']:.4f}"
        )

    print(
        f"checkpoint: {best_path}"
    )
    print(
        f"history:    {history_path}"
    )
    print(
        f"result:     {result_path}"
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Train one temporal BQN+BiGRU configuration "
            "on one frozen Schaefer100 split."
        )
    )

    ap.add_argument(
        "--temporal-h5",
        default=(
            "derivatives/"
            "abide_schaefer100_tr2_w44_s22_temporalfc.h5"
        ),
    )

    ap.add_argument(
        "--split",
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    # BQN architecture hyperparameters.
    # For clean temporal comparisons, use the already selected static-BQN
    # hyperparameters for that seed unless deliberately studying re-tuning.
    ap.add_argument(
        "--bqn-layers",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
    )

    ap.add_argument(
        "--bqn-dropout",
        type=float,
        choices=[0.0, 0.1, 0.2, 0.3],
        required=True,
    )

    ap.add_argument(
        "--bqn-activation",
        choices=[
            "gelu",
            "leaky_relu",
            "elu",
        ],
        required=True,
    )

    ap.add_argument(
        "--cluster-num",
        type=int,
        default=4,
    )

    # Temporal-specific tuning dimensions.
    ap.add_argument(
        "--bigru-hidden-dim",
        type=int,
        choices=[32, 64],
        default=32,
    )

    ap.add_argument(
        "--bigru-num-layers",
        type=int,
        choices=[1, 2],
        default=1,
    )

    ap.add_argument(
        "--bigru-dropout",
        type=float,
        choices=[0.0, 0.1, 0.2, 0.3],
        default=0.1,
    )

    ap.add_argument(
        "--attention-hidden-dim",
        type=int,
        choices=[32, 64],
        default=32,
    )

    ap.add_argument(
        "--classifier-hidden-dim",
        type=int,
        choices=[32, 64],
        default=64,
    )

    ap.add_argument(
        "--classifier-dropout",
        type=float,
        choices=[0.0, 0.1, 0.2, 0.3],
        default=0.1,
    )

    layer_norm_group = ap.add_mutually_exclusive_group()

    layer_norm_group.add_argument(
        "--temporal-layer-norm",
        dest="temporal_layer_norm",
        action="store_true",
    )

    layer_norm_group.add_argument(
        "--no-temporal-layer-norm",
        dest="temporal_layer_norm",
        action="store_false",
    )

    ap.set_defaults(
        temporal_layer_norm=True
    )

    # Training.
    ap.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
    )

    ap.add_argument(
        "--lr-schedule",
        choices=[
            "cosine",
            "constant",
        ],
        default="cosine",
    )

    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    # GRU stability. Set 0 to disable.
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    # Optional regularization experiment; default off.
    ap.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
    )

    # Optional static-BQN initialization experiment.
    ap.add_argument(
        "--bqn-init-checkpoint",
        default=None,
        help=(
            "Optional best_model.pt from a matching static BQN run. "
            "Default: train temporal BQN backbone from scratch."
        ),
    )

    ap.add_argument(
        "--evaluate-test",
        action="store_true",
    )

    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from run_dir/last_model.pt. "
            "Useful for Slurm preemption/requeue."
        ),
    )

    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--device",
        default=None,
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "results/"
            "temporal_bqn_bigru"
        ),
    )

    return ap.parse_args()


def main():
    args = parse_args()
    train_run(
        args
    )


if __name__ == "__main__":
    main()
