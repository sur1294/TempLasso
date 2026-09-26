#!/usr/bin/env python3
"""
ADNI (Schaefer-100) -> the two artifacts the unified pipeline consumes
======================================================================

Input is the collaborator's per-subject HDF5 (``adni_data_with_TR.h5``):

    <subject_id>/                       group attrs: label, TR_s, age, sex
        schaefer100/timeseries          (T, 100)  native TR
        schaefer100/corr_matrix         (100, 100)

Outputs (all in --out-dir, prefix ``adni_schaefer100``):

1. ``*_static.h5``  -- dense static FC in the layout ``StaticABIDEDataset``
   reads directly: ``subject_ids``, ``labels``, ``fc`` [N,R,R] plus
   ``tr_native``, ``age``, ``sex`` for audit. ``fc`` is the collaborator's
   ``corr_matrix`` as delivered (whole scan), i.e. the same artifact her
   harness trains on.
2. ``*_static_trunc.h5`` -- same layout, but ``fc`` recomputed as the
   Ledoit-Wolf correlation of the resampled, duration-truncated series
   (identical estimator to the temporal windows and to the ABIDE static
   file). Exists because whole-scan FC from a 591 s scan is less noisy than
   from a 293 s scan, and in ADNI scan duration is a proxy for the label.
3. ``*_tr{tr}_timeseries.h5`` -- Step-1 layout (``ts_resampled``,
   ``offsets_resampled``, ``subject_id``, ``label``, ``tr_native``,
   ``folder`` + attrs ``tr_common``/``n_rois``/``atlas``) so that
   ``build_step2_temporal_fc.py`` runs on it UNCHANGED.

Why this file exists at all
---------------------------
ADNI's acquisition is confounded with its label:

    TR 0.607 s : 102 subjects, 79 control / 23 patient, ~293 s scans
    TR 0.78  s :  10 subjects,  1 control /  9 patient, ~292 s scans
    TR 3.0   s :  26 subjects,  0 control / 26 patient, ~591 s scans

Every TR = 3.0 subject is a patient, and their scans are twice as long. A
temporal model therefore sees the label in the number of windows K
(~25 vs ~12) before it sees any connectivity. ``--max-duration-sec`` cuts
every subject to one common duration BEFORE windowing so K is constant, and
the split generator can stratify on TR group so no test fold is
accidentally enriched in the TR = 3.0 patients. Neither removes the
confound from the FC values themselves -- that has to be reported, which is
why the TR group is carried through into every artifact.

Design invariants shared with the ABIDE pipeline
------------------------------------------------
* Resampling keeps the endpoint (``arange(n_new) * tr_common``).
* Ragged storage: one concatenated 2-D array + ``offsets``.
* Missing metadata is a hard error, never a default.
* Nothing fold-dependent (ComBat, standardisation) is baked in.

Usage
-----
    python build_adni_artifacts.py \
        --in-h5 ~/Desktop/temporal-brain-network/TDA/adni_data_with_TR.h5 \
        --out-dir derivatives/adni --tr-common 2.0 --max-duration-sec 290
    python build_step2_temporal_fc.py \
        --in-h5 derivatives/adni/adni_schaefer100_tr2_timeseries.h5 \
        --out-dir derivatives/adni --window-sec 44 --stride-sec 22
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_step1_timeseries import native_k, resample_to_common_grid  # noqa: E402
from build_step2_temporal_fc import cov_to_corr  # noqa: E402

TR_GROUP_FAST_MAX = 1.0  # TR < 1.0 s -> "fast" (0.607 / 0.78), else "slow" (3.0)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return "unknown"


def tr_group(tr_s: float) -> str:
    return "fast" if tr_s < TR_GROUP_FAST_MAX else "slow"


def ledoit_wolf_corr(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf covariance -> correlation, same estimator as Step 2."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(x)
    corr, degenerate = cov_to_corr(lw.covariance_)
    if degenerate.any():
        raise ValueError(f"{int(degenerate.sum())} degenerate ROI(s) in a truncated series")
    return corr.astype(np.float32), float(lw.shrinkage_)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--atlas", default="schaefer100")
    ap.add_argument("--n-rois", type=int, default=100)
    ap.add_argument("--tr-common", type=float, default=2.0)
    ap.add_argument("--max-duration-sec", type=float, default=None,
                    help="Truncate every resampled series to this duration "
                         "(seconds) so window count K is identical across TR "
                         "groups. None keeps native duration.")
    ap.add_argument("--window-sec", type=float, default=44.0)
    ap.add_argument("--stride-sec", type=float, default=22.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"adni_{args.atlas}"
    tr_tag = f"tr{args.tr_common:g}".replace(".", "p")

    rows = []
    fc_whole, fc_trunc, res_list, off = [], [], [], [0]
    lam_trunc = []
    with h5py.File(args.in_h5, "r") as f:
        subjects = sorted(f.keys())
        for sid in subjects:
            g = f[sid]
            for key in ("label", "TR_s"):
                if key not in g.attrs:
                    raise SystemExit(f"{sid}: missing group attribute {key!r}")
            if args.atlas not in g:
                raise SystemExit(f"{sid}: no {args.atlas} group")
            ts = np.asarray(g[args.atlas]["timeseries"], dtype=np.float64)
            if ts.shape[1] != args.n_rois and ts.shape[0] == args.n_rois:
                ts = ts.T
            if ts.shape[1] != args.n_rois:
                raise SystemExit(f"{sid}: timeseries shape {ts.shape} != (T,{args.n_rois})")
            if not np.isfinite(ts).all():
                raise SystemExit(f"{sid}: non-finite timeseries")
            corr = np.asarray(g[args.atlas]["corr_matrix"], dtype=np.float32)
            if corr.shape != (args.n_rois, args.n_rois) or not np.isfinite(corr).all():
                raise SystemExit(f"{sid}: bad corr_matrix")

            tr = float(g.attrs["TR_s"])
            label = int(g.attrs["label"])
            if label not in (0, 1):
                raise SystemExit(f"{sid}: label {label} not binary")

            res = resample_to_common_grid(ts, tr, args.tr_common)
            n_native_frames = res.shape[0]
            if args.max_duration_sec is not None:
                keep = int(np.floor(args.max_duration_sec / args.tr_common)) + 1
                if res.shape[0] < keep:
                    raise SystemExit(
                        f"{sid}: only {res.shape[0]*args.tr_common:.0f}s after "
                        f"resampling, shorter than --max-duration-sec {args.max_duration_sec}"
                    )
                res = res[:keep]
            corr_t, lam = ledoit_wolf_corr(res)

            rows.append(dict(
                subject_id=sid, label=label, tr_native=tr, tr_group=tr_group(tr),
                age=float(g.attrs.get("age", np.nan)), sex=int(g.attrs.get("sex", -1)),
                n_frames_native=int(ts.shape[0]),
                n_frames_resampled_full=int(n_native_frames),
                n_frames_resampled=int(res.shape[0]),
                duration_sec=float((res.shape[0] - 1) * args.tr_common),
                K=native_k(res.shape[0], args.tr_common, args.window_sec, args.stride_sec),
                lw_shrinkage_trunc=lam,
            ))
            fc_whole.append(corr)
            fc_trunc.append(corr_t)
            lam_trunc.append(lam)
            res_list.append(res.astype(np.float32))
            off.append(off[-1] + res.shape[0])

    n = len(rows)
    ids = np.array([r["subject_id"] for r in rows], dtype="S")
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    tr_native = np.array([r["tr_native"] for r in rows], dtype=np.float64)
    tr_groups = np.array([r["tr_group"] for r in rows], dtype="S")
    Ks = np.array([r["K"] for r in rows], dtype=np.int64)

    provenance = dict(
        source=os.path.abspath(args.in_h5), atlas=args.atlas, n_rois=args.n_rois,
        tr_common=args.tr_common, max_duration_sec=args.max_duration_sec,
        window_sec=args.window_sec, stride_sec=args.stride_sec,
        git_commit=git_commit(), created=datetime.datetime.now().isoformat(timespec="seconds"),
        note=("ADNI acquisition is label-confounded (all TR=3.0 subjects are patients); "
              "tr_group is carried so splits can stratify on it and reports can control for it."),
    )

    def write_static(path, fcs, estimator):
        with h5py.File(path, "w") as o:
            o.create_dataset("subject_ids", data=ids)
            o.create_dataset("labels", data=labels)
            o.create_dataset("fc", data=np.stack(fcs).astype(np.float32))
            o.create_dataset("tr_native", data=tr_native)
            o.create_dataset("tr_group", data=tr_groups)
            o.create_dataset("age", data=np.array([r["age"] for r in rows]))
            o.create_dataset("sex", data=np.array([r["sex"] for r in rows]))
            for k, v in provenance.items():
                o.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else (v if v is not None else "none")
            o.attrs["fc_estimator"] = estimator

    static_whole = os.path.join(args.out_dir, f"{tag}_static.h5")
    static_trunc = os.path.join(args.out_dir, f"{tag}_{tr_tag}_static_trunc.h5")
    write_static(static_whole, fc_whole, "collaborator corr_matrix, whole scan, as delivered")
    write_static(static_trunc, fc_trunc,
                 f"Ledoit-Wolf correlation on resampled series (tr_common={args.tr_common}, "
                 f"max_duration_sec={args.max_duration_sec})")

    ts_path = os.path.join(args.out_dir, f"{tag}_{tr_tag}_timeseries.h5")
    with h5py.File(ts_path, "w") as o:
        o.create_dataset("ts_resampled", data=np.concatenate(res_list, axis=0), compression="gzip")
        o.create_dataset("offsets_resampled", data=np.array(off, dtype=np.int64))
        o.create_dataset("subject_id", data=ids)
        o.create_dataset("folder", data=ids)
        o.create_dataset("label", data=labels)
        o.create_dataset("tr_native", data=tr_native)
        o.create_dataset("tr_group", data=tr_groups)
        o.create_dataset("K_expected", data=Ks)
        o.attrs["tr_common"] = args.tr_common
        o.attrs["n_rois"] = args.n_rois
        o.attrs["atlas"] = args.atlas
        for k, v in provenance.items():
            o.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else (v if v is not None else "none")

    meta_path = os.path.join(args.out_dir, f"{tag}_{tr_tag}_metadata.csv")
    import csv
    with open(meta_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ------------------------------------------------------------ report
    print(f"subjects: {n}   labels: control={int((labels==0).sum())} patient={int((labels==1).sum())}")
    print(f"tr_common={args.tr_common}s  max_duration={args.max_duration_sec}s  "
          f"window/stride={args.window_sec}/{args.stride_sec}s")
    from collections import Counter
    grp = Counter((r["tr_group"], r["label"]) for r in rows)
    print("tr_group x label:", dict(sorted(grp.items())))
    print("K per subject   :", dict(sorted(Counter(Ks.tolist()).items())),
          "(constant across TR groups)" if len(set(Ks.tolist())) == 1 else "  <-- NOT constant: K leaks TR group")
    print("duration (s)    :", dict(sorted(Counter(round(r["duration_sec"]) for r in rows).items())))
    print(f"LW shrinkage (trunc) by tr_group: "
          + ", ".join(f"{g}={np.mean([r['lw_shrinkage_trunc'] for r in rows if r['tr_group']==g]):.3f}"
                      for g in ("fast", "slow")))
    print("wrote:", static_whole); print("wrote:", static_trunc); print("wrote:", ts_path); print("wrote:", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
