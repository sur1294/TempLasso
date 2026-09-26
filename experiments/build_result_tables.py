#!/usr/bin/env python3
"""
Build the final tables from the per-seed result CSVs (no training, no cluster).

Inputs (all under --results, default ``results/``):
    grid/<dataset>_<row>[_<backbone>]_selected.csv
        one line per seed: the validation-selected candidate of that grid and
        its single test evaluation (test_auc, test_accuracy, test_sensitivity,
        test_specificity). Written by ``experiments/run_grid.py select``.
    late_fusion_tuned_v2/<backbone>_<dataset>[_tempx].csv
        late fusion (probability average, ``fusion == "avg"``) of the two
        selected branches, with test_auc / test_acc / test_sensitivity /
        test_specificity. Written by ``experiments/late_fusion_from_checkpoints.py``.

Outputs:
    sheets/metrics_all_rows_per_seed.tsv   204 rows = 51 cells x {AUC, ACC, SEN, SPE}
    metrics_all_rows.md                    the same as mean ± std tables
    sheets/5runs-<DATASET>.tsv             AUC-only per-dataset sheets, one block per backbone

Conventions (identical to the delivered files):
    * values are x100, per-seed values printed with 1 decimal, mean/std with 2;
    * std is the population std (ddof = 0) over the 5 seeds;
    * seeds are ordered 42, 1042, 2042, 3042, 4042;
    * PPMI is labelled "PPMI (macro)": 4-class macro one-vs-rest AUC, top-1
      accuracy, macro recall (SEN) and macro specificity (SPE).

NOTE on the 5runs sheets: the versions first sent to the collaborator used the
collaborator-pipeline BQN numbers for the BQN X / TempX / X+TempX rows. This
script writes every row from OUR pipeline (the decision of 2026-09-25), so the
BQN block here differs from those first sheets; ``metrics_all_rows_per_seed.tsv``
was always our-pipeline only and is unchanged.

    python experiments/build_result_tables.py                 # write files
    python experiments/build_result_tables.py --check         # only verify the TSV is reproduced
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

SEEDS = [42, 1042, 2042, 3042, 4042]
DATASETS = (("abide", "ABIDE"), ("adni", "ADNI"), ("ppmi", "PPMI (macro)"))
BACKBONES = (("bqn", "BQN"), ("alter", "ALTER"), ("brainnetcnn", "BrainNetCNN"), ("gcn", "GCN"))
METRICS = (("test_auc", "AUC"), ("test_accuracy", "ACC"), ("test_sensitivity", "SEN"), ("test_specificity", "SPE"))


def grid_rows(base: Path, name: str) -> dict[int, dict]:
    with open(base / "grid" / f"{name}_selected.csv") as fh:
        return {int(r["seed"]): r for r in csv.DictReader(fh)}


def fusion_rows(base: Path, name: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(base / "late_fusion_tuned_v2" / f"{name}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["fusion"] != "avg":
                continue
            rows[int(r["seed"])] = {
                "test_auc": r["test_auc"],
                "test_accuracy": r["test_acc"],
                "test_sensitivity": r["test_sensitivity"],
                "test_specificity": r["test_specificity"],
            }
    return rows


def values(rows: dict[int, dict], key: str) -> list[float]:
    return [float(rows[s][key]) * 100 for s in SEEDS]


def mean_sd(v: list[float]) -> str:
    return f"{st.fmean(v):.2f} ± {st.pstdev(v):.2f}"


def cells(base: Path, ds: str):
    """Yield (backbone label, row label, rows) for one dataset in table order."""
    for bb, bn in BACKBONES:
        yield bn, "X", grid_rows(base, f"{ds}_x_{bb}")
        yield bn, "TempX", grid_rows(base, f"{ds}_tempx_{bb}")
        yield bn, "X+TempLasso", fusion_rows(base, f"{bb}_{ds}")
        yield bn, "X+TempX", fusion_rows(base, f"{bb}_{ds}_tempx")
    yield "—", "TempLasso", grid_rows(base, f"{ds}_templasso")


def build(base: Path) -> tuple[str, str, dict[str, str]]:
    tsv = ["Dataset\tBackbone\tRow\tMetric\tMean ± SD\tSeed 42\tSeed 1042\tSeed 2042\tSeed 3042\tSeed 4042"]
    md: list[str] = []
    sheets: dict[str, str] = {}
    for ds, name in DATASETS:
        md.append(f"\n**{name}**\n\n| Backbone | Row | AUC | ACC | SEN | SPE |\n|---|---|---|---|---|---|")
        sheet: list[str] = []
        current = None
        for bn, row, rows in cells(base, ds):
            md.append(f"| {bn} | {row} | " + " | ".join(mean_sd(values(rows, k)) for k, _ in METRICS) + " |")
            for k, lab in METRICS:
                v = values(rows, k)
                tsv.append(f"{name}\t{bn}\t{row}\t{lab}\t{mean_sd(v)}\t" + "\t".join(f"{x:.1f}" for x in v))
            block = bn if bn != "—" else "TempLasso (backbone-free)"
            if block != current:
                if current is not None:
                    sheet.append("")
                sheet.append(block)
                sheet.append("Dataset\tModel\tMean ± SD\tSeed 42\tSeed 1042\tSeed 2042\tSeed 3042\tSeed 4042")
                current = block
            v = values(rows, "test_auc")
            sheet.append(f"{name.split(' ')[0]}\t{row}\t{mean_sd(v)}\t" + "\t".join(f"{x:.1f}" for x in v))
        sheets[f"5runs-{name.split(' ')[0]}.tsv"] = "\n".join(sheet) + "\n"
    header = (
        "# Test AUC / ACC / SEN / SPE (x100), mean ± std over 5 seeds\n\n"
        "Selection on val loss; test once per seed; ACC/SEN/SPE at threshold 0.5 "
        "(PPMI: argmax, macro recall / macro specificity). Per-seed values: "
        "sheets/metrics_all_rows_per_seed.tsv\n"
    )
    return "\n".join(tsv) + "\n", header + "\n".join(md) + "\n", sheets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--check", action="store_true", help="compare against the existing TSV instead of writing")
    args = ap.parse_args()
    base = args.results
    tsv, md, sheets = build(base)
    target = base / "sheets" / "metrics_all_rows_per_seed.tsv"
    if args.check:
        existing = target.read_text()
        print("metrics_all_rows_per_seed.tsv reproduced byte-for-byte:", existing == tsv)
        if existing != tsv:
            raise SystemExit(1)
        return
    (base / "sheets").mkdir(exist_ok=True)
    target.write_text(tsv)
    (base / "metrics_all_rows.md").write_text(md)
    for name, text in sheets.items():
        (base / "sheets" / name).write_text(text)
    print("wrote", target, base / "metrics_all_rows.md", *[base / "sheets" / n for n in sheets])


if __name__ == "__main__":
    main()
