#!/usr/bin/env python3
"""
Frozen 70/10/20 splits for a dataset other than ABIDE, with the SAME
procedure as ``make_abide_splits.py`` (two nested StratifiedShuffleSplits,
seeds 42/1042/2042/3042/4042, keyed by subject ID) plus one extension:
``--stratify`` chooses the stratification key.

    label       stratify on the class label only (ABIDE procedure)
    label_tr    stratify on (tr_group, label) jointly

The second exists for ADNI, whose acquisition TR is confounded with the
label (every TR = 3.0 subject is a patient). Stratifying on label alone lets
the test fold's share of TR = 3.0 patients drift from seed to seed, which
would move the AUC for reasons that have nothing to do with the model.

Outputs ``<prefix>_split_seed<seed>.npz`` with ``train_ids``, ``val_ids``,
``test_ids`` (strings) and a scalar ``seed`` -- exactly what
``unified_bn.data.load_frozen_split`` and ``ABIDERuntimeFactory`` expect --
plus ``split_summary.json``.

Nothing here reads FC values; only IDs, labels and the stratification key.
Regenerating an existing split directory is refused unless --force is given,
because reported numbers depend on frozen splits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_abide_splits import make_split, verify_split  # noqa: E402

SEEDS = (42, 1042, 2042, 3042, 4042)


def _ids(arr) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static-h5", required=True, help="dense static file (subject_ids, labels, tr_group)")
    ap.add_argument("--temporal-h5", required=True, help="Step-2 file; cohort must match exactly")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. adni")
    ap.add_argument("--stratify", choices=("label", "label_tr"), default="label")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with h5py.File(args.static_h5, "r") as s:
        ids = _ids(s["subject_ids"][:]); labels = s["labels"][:].astype(np.int64)
        tr_group = _ids(s["tr_group"][:]) if "tr_group" in s else None
    with h5py.File(args.temporal_h5, "r") as t:
        t_ids = _ids(t["subject_id"][:]); t_labels = t["label"][:].astype(np.int64)

    if set(ids) != set(t_ids):
        raise SystemExit(f"cohort mismatch: static {len(ids)} vs temporal {len(t_ids)} "
                         f"(only-static={sorted(set(ids)-set(t_ids))[:5]}, only-temporal={sorted(set(t_ids)-set(ids))[:5]})")
    t_lab = dict(zip(t_ids, t_labels))
    bad = [i for i, l in zip(ids, labels) if t_lab[i] != l]
    if bad:
        raise SystemExit(f"label disagreement for {len(bad)} subjects, e.g. {bad[:5]}")

    if args.stratify == "label_tr":
        if tr_group is None:
            raise SystemExit("--stratify label_tr needs a tr_group dataset in the static file")
        strat = np.array([f"{g}|{l}" for g, l in zip(tr_group, labels)])
    else:
        strat = labels.copy()

    os.makedirs(args.out_dir, exist_ok=True)
    existing = [f for f in os.listdir(args.out_dir) if f.startswith(f"{args.prefix}_split_seed")]
    if existing and not args.force:
        raise SystemExit(f"{args.out_dir} already has {existing}; refusing to regenerate frozen splits "
                         f"(pass --force only if these were never used for a reported number)")

    order = np.argsort(ids)  # deterministic order independent of file order
    ids, labels, strat = ids[order], labels[order], strat[order]
    tr_group = tr_group[order] if tr_group is not None else None

    summary = {"prefix": args.prefix, "n": int(len(ids)), "stratify": args.stratify,
               "seeds": list(SEEDS), "static_h5": os.path.abspath(args.static_h5),
               "temporal_h5": os.path.abspath(args.temporal_h5), "splits": {}}
    for seed in SEEDS:
        tr_idx, va_idx, te_idx = make_split(ids, strat, seed)
        verify_split(ids[tr_idx], ids[va_idx], ids[te_idx], ids)
        np.savez_compressed(
            os.path.join(args.out_dir, f"{args.prefix}_split_seed{seed}.npz"),
            train_ids=ids[tr_idx], val_ids=ids[va_idx], test_ids=ids[te_idx],
            seed=np.array(seed, dtype=np.int64),
        )
        def desc(idx):
            d = {"n": int(len(idx)), "control": int((labels[idx] == 0).sum()), "patient": int((labels[idx] == 1).sum())}
            if tr_group is not None:
                d["slow_patient"] = int(((tr_group[idx] == "slow") & (labels[idx] == 1)).sum())
            return d
        summary["splits"][str(seed)] = {"train": desc(tr_idx), "val": desc(va_idx), "test": desc(te_idx)}
        print(f"seed {seed}: train {desc(tr_idx)} | val {desc(va_idx)} | test {desc(te_idx)}")
    with open(os.path.join(args.out_dir, "split_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
