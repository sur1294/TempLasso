#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from bqn_dataset import build_loaders
from bqn_model import BQNConfig, StaticBQN
from train_bqn import evaluate, fix_seed

SEEDS = [42, 1042, 2042, 3042, 4042]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="derivatives/abide_data_tr.h5")
    ap.add_argument("--split-dir", default="splits")
    ap.add_argument("--selected-dir", default="results/bqn_selected")
    ap.add_argument("--out-dir", default="results/bqn_final")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        sel_path = Path(args.selected_dir) / f"selected_seed{seed}.json"
        if not sel_path.exists():
            raise FileNotFoundError(sel_path)
        with sel_path.open() as f:
            sel = json.load(f)
        if int(sel["seed"]) != seed:
            raise RuntimeError(f"{sel_path}: seed mismatch")

        ckpt_path = Path(sel["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        split_path = Path(args.split_dir) / f"abide_split_seed{seed}.npz"
        if not split_path.exists():
            raise FileNotFoundError(split_path)

        fix_seed(seed)
        _, _, test_loader = build_loaders(
            h5_path=args.h5,
            split_path=str(split_path),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_seed=seed,
        )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("model_config")
        if cfg_dict is None:
            raise RuntimeError(f"{ckpt_path}: checkpoint missing model_config")

        cfg = BQNConfig(**cfg_dict)
        if int(cfg.n_layers) != int(sel["layers"]):
            raise RuntimeError("Selected layers disagree with checkpoint")
        if str(cfg.activation) != str(sel["activation"]):
            raise RuntimeError("Selected activation disagrees with checkpoint")
        if not np.isclose(float(cfg.dropout), float(sel["dropout"])):
            raise RuntimeError("Selected dropout disagrees with checkpoint")

        model = StaticBQN(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        test_metrics = evaluate(model, test_loader, device)

        result = {
            "seed": seed,
            "selection_rule": "minimum validation cross-entropy loss",
            "selected_config": {
                "layers": int(sel["layers"]),
                "activation": str(sel["activation"]),
                "dropout": float(sel["dropout"]),
                "best_epoch": int(sel["best_epoch"]),
                "val_loss": float(sel["val_loss"]),
            },
            "checkpoint": str(ckpt_path),
            "split_file": str(split_path),
            "test": test_metrics,
        }

        out_path = out / f"test_seed{seed}.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

        print(
            f"Seed {seed}: AUC={test_metrics['auc']:.4f} "
            f"ACC={test_metrics['acc']:.4f} "
            f"SEN={test_metrics['sens']:.4f} "
            f"SPE={test_metrics['spec']:.4f}"
        )

if __name__ == "__main__":
    main()
