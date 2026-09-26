#!/usr/bin/env python3
"""
Train ONE static BQN configuration on ONE frozen ABIDE Schaefer100 split.

This file is intentionally narrow:
- one frozen seed
- one BQN hyperparameter configuration
- no hyperparameter search
- no temporal data
- no new random split
- no feature fitting / normalization
- no mixup
- no class weighting
- no gradient clipping
- no BatchNorm added outside the original BQN model
- no early stopping

The model is trained for the full requested epoch budget.  The checkpoint
with the MINIMUM VALIDATION LOSS is retained.  Test evaluation is performed
only if --evaluate-test is explicitly supplied, and only AFTER the best
validation checkpoint has been restored.

The intended later workflow is:
    tune_bqn.py
        -> call training WITHOUT test evaluation for candidate configs
        -> select configuration using validation only
        -> retrain/evaluate the selected configuration on test

Atlas Matters protocol used here
--------------------------------
ABIDE Schaefer100
Frozen 70/10/20 subject-ID split
Batch size 16
Adam
Initial LR 1e-4
Weight decay 1e-4
Up to 200 epochs
Checkpoint selection: minimum validation loss

The Atlas paper states that LR decays toward 1e-5 but does not, by itself,
uniquely specify the functional form of the decay.  This script therefore
makes the schedule explicit via --lr-schedule.  The default is cosine decay
to --lr-min=1e-5.  Record this choice in reported experiments.

Run from project root
---------------------
Validation-only run:
    python models/bqn/train_bqn.py \
        --h5 derivatives/abide_data_tr.h5 \
        --split splits/abide_split_seed42.npz \
        --seed 42 \
        --layers 3 \
        --dropout 0.1 \
        --activation leaky_relu \
        --out-dir results/bqn

Final selected-config run including test:
    python models/bqn/train_bqn.py \
        --h5 derivatives/abide_data_tr.h5 \
        --split splits/abide_split_seed42.npz \
        --seed 42 \
        --layers 3 \
        --dropout 0.1 \
        --activation leaky_relu \
        --out-dir results/bqn \
        --evaluate-test
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

from bqn_dataset import build_loaders, load_frozen_split
from bqn_model import BQNConfig, StaticBQN


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_seed(seed: int) -> None:
    """
    Set the random seeds used by Python, NumPy, and PyTorch.

    Deterministic algorithms are requested where possible.  warn_only=True
    avoids silently switching off determinism while still surfacing operations
    for which PyTorch cannot guarantee it.
    """
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
) -> Dict[str, float]:
    """
    Evaluate loss and binary-classification metrics.

    AUC is computed from P(class=1), not from hard predictions.
    Sensitivity is recall for label 1.
    Specificity is recall for label 0.
    """
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    n_total = 0

    logits_all = []
    labels_all = []

    for fc, labels, _subject_ids in loader:
        fc = fc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(fc)

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits encountered in evaluation.")

        total_loss += float(criterion(logits, labels).item())
        n_total += int(labels.numel())

        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    if n_total == 0:
        raise RuntimeError("Evaluation loader is empty.")

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0).numpy()

    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    predictions = torch.argmax(logits, dim=1).numpy()

    return {
        "loss": total_loss / n_total,
        "auc": float(roc_auc_score(labels, probabilities)),
        "acc": float(accuracy_score(labels, predictions)),
        "sens": float(
            recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "spec": float(
            recall_score(
                labels,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
    }


# ---------------------------------------------------------------------------
# LR scheduling
# ---------------------------------------------------------------------------

def set_epoch_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    base_lr: float,
    lr_min: float,
    schedule: str,
) -> float:
    """
    Set LR at the beginning of an epoch.

    The Atlas manuscript specifies decay toward 1e-5 but does not uniquely
    identify the functional form.  We therefore make the implementation
    explicit and save it in the run metadata.

    cosine:
        epoch 0 starts at base_lr and the final epoch approaches lr_min.

    constant:
        no decay; useful only for controlled ablations.
    """
    if schedule == "cosine":
        if epochs <= 1:
            lr = lr_min
        else:
            progress = epoch / (epochs - 1)
            lr = lr_min + 0.5 * (base_lr - lr_min) * (
                1.0 + math.cos(math.pi * progress)
            )
    elif schedule == "constant":
        lr = base_lr
    else:
        raise ValueError(f"Unknown LR schedule: {schedule}")

    for group in optimizer.param_groups:
        group["lr"] = lr

    return float(lr)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Standard supervised BQN training.

    No mixup, no class weights, and no gradient clipping are introduced here.
    """
    model.train()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    n_total = 0

    for fc, labels, _subject_ids in loader:
        fc = fc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(fc)

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits encountered in training.")

        loss_mean = nn.functional.cross_entropy(
            logits,
            labels,
            reduction="mean",
        )

        if not torch.isfinite(loss_mean):
            raise FloatingPointError("Non-finite training loss encountered.")

        loss_mean.backward()
        optimizer.step()

        with torch.no_grad():
            batch_sum = nn.functional.cross_entropy(
                logits,
                labels,
                reduction="sum",
            ).item()

        total_loss += float(batch_sum)
        n_total += int(labels.numel())

    if n_total == 0:
        raise RuntimeError("Training loader is empty.")

    return total_loss / n_total


def save_history_csv(history, path: Path) -> None:
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_auc",
        "val_acc",
        "val_sens",
        "val_spec",
        "is_best",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def build_run_directory(args) -> Path:
    """
    Give every seed/configuration its own directory so tuning runs never
    overwrite each other.
    """
    config_name = (
        f"seed{args.seed}"
        f"_L{args.layers}"
        f"_{args.activation}"
        f"_drop{args.dropout:g}"
    )

    run_dir = Path(args.out_dir) / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_run(args) -> Dict:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0 or args.lr_min <= 0:
        raise ValueError("Learning rates must be positive.")
    if args.lr_min > args.lr:
        raise ValueError("--lr-min cannot exceed --lr.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be nonnegative.")

    # ------------------------------------------------------------
    # Frozen split metadata
    # ------------------------------------------------------------
    split = load_frozen_split(args.split)

    if "seed" not in split:
        raise RuntimeError(
            "Frozen split NPZ has no stored seed. "
            "Refusing to infer which split this is."
        )

    split_seed = int(np.asarray(split["seed"]).item())

    if split_seed != args.seed:
        raise ValueError(
            f"--seed={args.seed} but split file stores seed={split_seed}. "
            "Use the matching frozen split file."
        )

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    fix_seed(args.seed)

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    train_loader, val_loader, test_loader = build_loaders(
        h5_path=args.h5,
        split_path=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_seed=args.seed,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    config = BQNConfig(
        n_rois=100,
        n_layers=args.layers,
        dropout=args.dropout,
        activation=args.activation,
        cluster_num=args.cluster_num,
        preserve_unused_mlp_b=True,
        orthogonal_centers=True,
        freeze_centers=True,
        project_assignment=True,
    )

    model = StaticBQN(config)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = build_run_directory(args)
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.csv"
    result_path = run_dir / "result.json"

    # Save run metadata before training.
    metadata = {
        "static_h5": str(Path(args.h5)),
        "split_file": str(Path(args.split)),
        "split_seed": split_seed,
        "model_seed": args.seed,
        "model_config": asdict(config),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "base_lr": args.lr,
            "lr_min": args.lr_min,
            "lr_schedule": args.lr_schedule,
            "weight_decay": args.weight_decay,
            "checkpoint_rule": "minimum validation cross-entropy loss",
            "early_stopping": False,
            "mixup": False,
            "class_weights": False,
            "gradient_clipping": False,
            "fc_transform": "none; use stored corr_matrix directly",
            "test_evaluated": bool(args.evaluate_test),
        },
        "device": str(device),
        "parameter_summary": model.parameter_summary(),
    }

    with (run_dir / "run_config.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------
    # Full training budget, select best validation-loss checkpoint
    # ------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    best_val_metrics: Optional[Dict[str, float]] = None

    history = []

    print("=" * 78)
    print("STATIC BQN TRAINING")
    print("=" * 78)
    print(f"device:      {device}")
    print(f"seed:        {args.seed}")
    print(f"split:       {args.split}")
    print(f"layers:      {args.layers}")
    print(f"activation:  {args.activation}")
    print(f"dropout:     {args.dropout}")
    print(f"clusters:    {args.cluster_num}")
    print(f"epochs:      {args.epochs}")
    print(f"batch size:  {args.batch_size}")
    print(f"LR:          {args.lr:g} -> {args.lr_min:g} ({args.lr_schedule})")
    print(f"weight decay:{args.weight_decay:g}")
    print(f"parameters:  {model.parameter_summary()}")
    print(f"test now:    {args.evaluate_test}")
    print("=" * 78)

    start_time = time.time()

    for epoch in range(args.epochs):
        lr = set_epoch_lr(
            optimizer=optimizer,
            epoch=epoch,
            epochs=args.epochs,
            base_lr=args.lr,
            lr_min=args.lr_min,
            schedule=args.lr_schedule,
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        is_best = val_metrics["loss"] < best_val_loss

        if is_best:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch + 1
            best_val_metrics = dict(val_metrics)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config),
                    "seed": args.seed,
                    "split_file": str(args.split),
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_metrics,
                },
                checkpoint_path,
            )

        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_acc": val_metrics["acc"],
            "val_sens": val_metrics["sens"],
            "val_spec": val_metrics["spec"],
            "is_best": int(is_best),
        }
        history.append(row)

        print(
            f"epoch {epoch + 1:03d}/{args.epochs} "
            f"lr={lr:.3e} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
            + ("  *BEST*" if is_best else "")
        )

    elapsed = time.time() - start_time

    if best_epoch is None or best_val_metrics is None:
        raise RuntimeError("No best validation checkpoint was saved.")

    save_history_csv(history, history_path)

    # ------------------------------------------------------------
    # Reload selected checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Re-evaluate val to verify checkpoint integrity.
    restored_val_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
    )

    if not np.isclose(
        restored_val_metrics["loss"],
        best_val_metrics["loss"],
        rtol=1e-6,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Restored checkpoint validation loss does not match saved best "
            f"value: restored={restored_val_metrics['loss']}, "
            f"saved={best_val_metrics['loss']}"
        )

    # ------------------------------------------------------------
    # Optional test evaluation
    # ------------------------------------------------------------
    test_metrics = None

    if args.evaluate_test:
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
        )

    result = {
        "seed": args.seed,
        "split_file": str(args.split),
        "best_epoch": best_epoch,
        "best_val": restored_val_metrics,
        "test": test_metrics,
        "elapsed_seconds": elapsed,
        "model_config": asdict(config),
        "training": metadata["training"],
        "parameter_summary": model.parameter_summary(),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    with result_path.open("w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)
    print(f"best epoch: {best_epoch}")
    print(
        "best val: "
        f"loss={restored_val_metrics['loss']:.5f} "
        f"auc={restored_val_metrics['auc']:.4f} "
        f"acc={restored_val_metrics['acc']:.4f} "
        f"sens={restored_val_metrics['sens']:.4f} "
        f"spec={restored_val_metrics['spec']:.4f}"
    )

    if test_metrics is None:
        print("test: NOT EVALUATED")
        print(
            "This is the correct mode for hyperparameter tuning. "
            "Use --evaluate-test only after selecting the configuration "
            "from validation results."
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

    print(f"checkpoint: {checkpoint_path}")
    print(f"history:    {history_path}")
    print(f"result:     {result_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one static BQN config on one frozen Schaefer100 split."
    )

    parser.add_argument(
        "--h5",
        default="derivatives/abide_data_tr.h5",
        help="Old static ABIDE HDF5.",
    )

    parser.add_argument(
        "--split",
        required=True,
        help="Frozen split NPZ for this seed.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Must match the seed stored inside --split.",
    )

    # BQN/Atlas tuning dimensions.
    parser.add_argument(
        "--layers",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        choices=[0.0, 0.1, 0.2, 0.3],
        required=True,
    )

    parser.add_argument(
        "--activation",
        choices=["gelu", "leaky_relu", "elu"],
        required=True,
    )

    parser.add_argument(
        "--cluster-num",
        type=int,
        default=4,
        help="Original BQN-demo default; not part of the Atlas shared grid.",
    )

    # Atlas protocol.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--lr-schedule",
        choices=["cosine", "constant"],
        default="cosine",
        help=(
            "Atlas manuscript states decay toward 1e-5 but does not uniquely "
            "specify the curve. Default makes the choice explicit as cosine."
        ),
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Evaluate test only after restoring the minimum-val-loss "
            "checkpoint. Omit this flag during hyperparameter tuning."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="e.g. cuda, cuda:0, cpu. Default: CUDA if available.",
    )

    parser.add_argument(
        "--out-dir",
        default="results/bqn",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_run(args)


if __name__ == "__main__":
    main()
