#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

SEEDS = [42, 1042, 2042, 3042, 4042]
METRICS = ["auc", "acc", "sens", "spec"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/bqn_final")
    args = ap.parse_args()

    d = Path(args.results_dir)
    rows = []

    for seed in SEEDS:
        p = d / f"test_seed{seed}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open() as f:
            r = json.load(f)
        if int(r["seed"]) != seed:
            raise RuntimeError(f"{p}: seed mismatch")

        row = {
            "seed": seed,
            "layers": int(r["selected_config"]["layers"]),
            "activation": str(r["selected_config"]["activation"]),
            "dropout": float(r["selected_config"]["dropout"]),
            "best_epoch": int(r["selected_config"]["best_epoch"]),
        }
        for m in METRICS:
            row[m] = float(r["test"][m])
        rows.append(row)

    summary = {}
    for m in METRICS:
        vals = np.array([r[m] for r in rows], dtype=float)
        summary[m] = {
            "mean": float(vals.mean()),
            "std_sample": float(vals.std(ddof=1)),
            "std_population": float(vals.std(ddof=0)),
        }

    print("\nPer-seed results")
    print("=" * 72)
    for r in rows:
        print(
            f"seed={r['seed']:4d} L={r['layers']} act={r['activation']:10s} "
            f"drop={r['dropout']:.1f} AUC={r['auc']:.4f} ACC={r['acc']:.4f} "
            f"SEN={r['sens']:.4f} SPE={r['spec']:.4f}"
        )

    print("\nFive-seed summary")
    print("=" * 72)
    for m in METRICS:
        print(f"{m.upper():4s}: {summary[m]['mean']:.4f} ± {summary[m]['std_sample']:.4f}")

    fields = ["seed","layers","activation","dropout","best_epoch","auc","acc","sens","spec"]
    with (d / "final_per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with (d / "final_summary.json").open("w") as f:
        json.dump({
            "seeds": SEEDS,
            "per_seed": rows,
            "summary": summary,
            "reported_std_recommendation": "sample standard deviation across five seeds (ddof=1); verify historical manuscript convention if needed",
        }, f, indent=2)

if __name__ == "__main__":
    main()
