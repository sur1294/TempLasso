#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

SEEDS = [42, 1042, 2042, 3042, 4042]
EXPECTED = 60
RUN_RE = re.compile(
    r"^seed(?P<seed>\d+)_L(?P<layers>[1-5])_"
    r"(?P<activation>gelu|leaky_relu|elu)_drop"
    r"(?P<dropout>0(?:\.0)?|0\.1|0\.2|0\.3)$"
)

def load_json(p):
    with p.open() as f:
        return json.load(f)

def parse_dir(p):
    m = RUN_RE.match(p.name)
    if not m:
        raise ValueError(f"Unexpected run directory: {p.name}")
    return int(m["seed"]), int(m["layers"]), m["activation"], float(m["dropout"])

def collect(tuning_dir: Path, seed: int):
    rows, seen = [], set()
    for run_dir in sorted(tuning_dir.glob(f"seed{seed}_L*")):
        if not run_dir.is_dir():
            continue
        s, layers, act, drop = parse_dir(run_dir)
        if s != seed:
            raise RuntimeError(f"Seed mismatch in {run_dir}")
        key = (layers, act, drop)
        if key in seen:
            raise RuntimeError(f"Duplicate config for seed {seed}: {key}")
        seen.add(key)

        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        r = load_json(result_path)

        if int(r["seed"]) != seed:
            raise RuntimeError(f"{result_path}: result seed mismatch")
        if r.get("test") is not None:
            raise RuntimeError(f"{result_path}: tuning result already contains test metrics")

        bv = r.get("best_val")
        if not isinstance(bv, dict) or "loss" not in bv:
            raise RuntimeError(f"{result_path}: missing best_val.loss")

        mc = r.get("model_config", {})
        if int(mc.get("n_layers", -1)) != layers:
            raise RuntimeError(f"{result_path}: layer mismatch")
        if str(mc.get("activation")) != act:
            raise RuntimeError(f"{result_path}: activation mismatch")
        if float(mc.get("dropout", -1)) != drop:
            raise RuntimeError(f"{result_path}: dropout mismatch")

        rows.append({
            "seed": seed,
            "layers": layers,
            "activation": act,
            "dropout": drop,
            "best_epoch": int(r["best_epoch"]),
            "val_loss": float(bv["loss"]),
            "val_auc": float(bv["auc"]),
            "val_acc": float(bv["acc"]),
            "val_sens": float(bv["sens"]),
            "val_spec": float(bv["spec"]),
            "checkpoint": str(r["checkpoint"]),
            "result_json": str(result_path),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-dir", default="results/bqn_tuning")
    ap.add_argument("--out-dir", default="results/bqn_selected")
    args = ap.parse_args()

    tuning = Path(args.tuning_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    selected = []
    for seed in SEEDS:
        rows = collect(tuning, seed)
        if len(rows) != EXPECTED:
            raise RuntimeError(f"Seed {seed}: expected {EXPECTED} completed runs, found {len(rows)}")

        best = sorted(
            rows,
            key=lambda x: (x["val_loss"], x["layers"], x["dropout"], x["activation"])
        )[0]
        selected.append(best)

        with (out / f"selected_seed{seed}.json").open("w") as f:
            json.dump(best, f, indent=2)

        print(
            f"Seed {seed}: L={best['layers']} drop={best['dropout']} "
            f"act={best['activation']} epoch={best['best_epoch']} "
            f"val_loss={best['val_loss']:.6f} val_auc={best['val_auc']:.4f}"
        )

    fields = list(selected[0].keys())
    with (out / "selected_configs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)

    with (out / "selected_configs.json").open("w") as f:
        json.dump({
            "selection_rule": "minimum validation cross-entropy loss, independently per seed",
            "n_candidates_per_seed": EXPECTED,
            "selected": selected,
        }, f, indent=2)

if __name__ == "__main__":
    main()
