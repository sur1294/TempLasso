#!/usr/bin/env python3
"""
Step 1 - ABIDE Schaefer-100 time series -> common-clock, labelled artifact
=========================================================================

Reads per-subject parcellated ROI time series (.mat), attaches the
per-subject TR from the IDA export, resamples every subject onto one
shared temporal grid, and writes a single ragged HDF5 file plus a
metadata CSV.

Deliberately does NOT do: windowing, covariance/FC, ComBat, sparsify.
Those are later steps; ComBat and sparsify belong in the dataloader
because they are fold-dependent.

Usage
-----
    python build_step1_timeseries.py \
        --data-dir /groups/bcoskunuzer/sxk230046/fall2026/Brainnetwork/abide/abide \
        --csv      /groups/bcoskunuzer/sxk230046/fall2026/Brainnetwork/idaSearch_8_09_2026.csv \
        --out-dir  /groups/bcoskunuzer/sxk230046/fall2026/Brainnetwork/derivatives \
        --tr-common 2.0

Optional cohort cross-check against the static-branch npz:
        --npz /path/to/node_features_tda60_ssm6_clean.npz
"""

import argparse
import csv as csvmod
import datetime
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict

import numpy as np
from scipy.interpolate import CubicSpline

try:
    import h5py
except ImportError:
    sys.exit("h5py is required:  pip install h5py")

import scipy.io as sio


# ---------------------------------------------------------------------------
# .mat loading  (these files are MATLAB 5.0 / v7 with a single var 'data',
#  but the h5py branch is kept in case any subject was written as v7.3)
# ---------------------------------------------------------------------------

def load_mat_array(path):
    """Return the single numeric array stored in a .mat file, as float64."""
    try:
        m = sio.loadmat(path)
        keys = [k for k in m if not k.startswith("__")]
        if not keys:
            raise ValueError(f"no variables in {path}")
        if "data" in keys:
            arr = m["data"]
        elif len(keys) == 1:
            arr = m[keys[0]]
        else:
            raise ValueError(f"{path}: ambiguous variables {keys}")
        return np.asarray(arr, dtype=np.float64)
    except NotImplementedError:
        # MATLAB v7.3 -> HDF5
        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            key = "data" if "data" in keys else keys[0]
            # v7.3 stores column-major, so h5py hands it back transposed
            return np.asarray(f[key][()], dtype=np.float64).T


def orient_timeseries(arr, n_rois, subject):
    """
    Coerce a parcellated series to (T, R).  Raises if the orientation is
    genuinely ambiguous (T == R) or if neither axis matches n_rois.
    """
    if arr.ndim != 2:
        raise ValueError(f"{subject}: expected 2-D array, got shape {arr.shape}")
    r0, r1 = arr.shape
    if r0 == n_rois and r1 == n_rois:
        raise ValueError(
            f"{subject}: shape {arr.shape} is square at R={n_rois}; "
            "cannot tell (T,R) from (R,T). Inspect this subject by hand."
        )
    if r1 == n_rois:
        return arr                      # already (T, R)
    if r0 == n_rois:
        return arr.T                    # (R, T) -> (T, R)
    raise ValueError(
        f"{subject}: neither axis of {arr.shape} equals n_rois={n_rois}"
    )


# ---------------------------------------------------------------------------
# TR lookup from the IDA export
# ---------------------------------------------------------------------------

def load_tr_map(csv_path):
    """
    subject_id (int) -> TR in SECONDS.

    IDA reports TR in milliseconds in the 'Imaging Protocol' column
    (e.g. 'TR=1500.0'), so divide by 1000. Rows without a TR are skipped;
    a subject that ends up missing here is a hard error at build time
    rather than a silently-defaulted value.
    """
    tr = {}
    with open(csv_path, newline="") as fh:
        for row in csvmod.DictReader(fh):
            if row.get("Description", "").strip() != "Resting State fMRI":
                continue
            m = re.search(r"TR\s*=\s*([\d.]+)", row.get("Imaging Protocol", "") or "")
            if not m:
                continue
            sid = int(row["Subject ID"])
            val = float(m.group(1)) / 1000.0
            if sid in tr and abs(tr[sid] - val) > 1e-9:
                raise ValueError(f"subject {sid}: conflicting TRs {tr[sid]} vs {val}")
            tr[sid] = val
    return tr


# ---------------------------------------------------------------------------
# Step 0 - resample onto the common grid
# ---------------------------------------------------------------------------

def resample_to_common_grid(x, tr_orig, tr_common):
    """
    x        : (T, R) on the subject's native grid
    tr_orig  : subject TR, seconds
    tr_common: shared target TR, seconds
    returns  : (T', R) on the common grid

    The native time axis spans [0, (T-1)*tr_orig]. The new grid is built
    with arange(n_new)*tr_common where n_new = floor(span/tr_common) + 1,
    which KEEPS the endpoint. (np.arange(0, span, tr_common) excludes it
    and can silently cost a subject their final window.)
    """
    T, R = x.shape
    if T < 4:
        raise ValueError(f"only {T} timepoints; cubic spline needs >= 4")
    t_orig = np.arange(T) * tr_orig
    span = t_orig[-1]
    n_new = int(np.floor(span / tr_common)) + 1
    t_new = np.arange(n_new) * tr_common

    out = np.empty((n_new, R), dtype=np.float64)
    for r in range(R):
        out[:, r] = CubicSpline(t_orig, x[:, r])(t_new)
    return out


def native_k(n_frames, tr_common, window_sec, stride_sec):
    """How many windows this many frames supports (0 if fewer than one)."""
    w = round(window_sec / tr_common)
    step = round(stride_sec / tr_common)
    if w < 2 or step < 1:
        return 0
    return max(0, (n_frames - w) // step + 1)


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

SUBDIR_RE = re.compile(r"^sub-(control|patient)(\d+)$")

def discover_subjects(data_dir):
    """-> list of (dirname, subject_id:int, label:int)  sorted by id."""
    found = []
    for name in sorted(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, name)):
            continue
        m = SUBDIR_RE.match(name)
        if not m:
            print(f"  [skip] unrecognised directory name: {name}")
            continue
        found.append((name, int(m.group(2)), 0 if m.group(1) == "control" else 1))
    found.sort(key=lambda t: t[1])
    return found


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="directory containing sub-control*/ and sub-patient*/")
    ap.add_argument("--csv", required=True, help="idaSearch_*.csv with per-subject TR")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--atlas", default="schaefer100",
                    help="filename token, e.g. schaefer100 / AAL116 / harvard48")
    ap.add_argument("--n-rois", type=int, default=100)
    ap.add_argument("--tr-common", type=float, default=2.0,
                    help="shared grid TR in seconds (cohort mode is 2.0)")
    ap.add_argument("--window-sec", type=float, default=44.0,
                    help="only used to REPORT native K; no windowing is saved")
    ap.add_argument("--stride-sec", type=float, default=22.0)
    ap.add_argument("--npz", default=None,
                    help="optional node_features npz to cross-check the cohort")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- cohort -----------------------------------------------------------
    print(f"Scanning {args.data_dir} ...")
    subjects = discover_subjects(args.data_dir)
    print(f"  {len(subjects)} subject directories")
    if not subjects:
        sys.exit("no subjects found - check --data-dir")

    keep_ids = None
    if args.npz:
        ref = np.load(args.npz, allow_pickle=True)
        keep_ids = {int(re.sub(r"\D", "", s)) for s in ref["subjects"]}
        ref_lab = {int(re.sub(r"\D", "", s)): int(l)
                   for s, l in zip(ref["subjects"], ref["labels"])}
        on_disk = {sid for _, sid, _ in subjects}
        print(f"  npz cohort: {len(keep_ids)}")
        print(f"  on disk but not in npz (will be EXCLUDED): {len(on_disk - keep_ids)}")
        print(f"  in npz but not on disk (MISSING):          {len(keep_ids - on_disk)}")
        if keep_ids - on_disk:
            print(f"    {sorted(keep_ids - on_disk)[:20]}")
        subjects = [t for t in subjects if t[1] in keep_ids]
        # label agreement between folder name and npz
        bad = [sid for _, sid, lab in subjects if ref_lab.get(sid) != lab]
        if bad:
            print(f"  !! label mismatch (folder vs npz) for {len(bad)}: {bad[:20]}")
        else:
            print("  labels agree with npz: OK")
        print(f"  proceeding with {len(subjects)} subjects")

    # ---- TR ---------------------------------------------------------------
    tr_map = load_tr_map(args.csv)
    print(f"\nTR entries in CSV: {len(tr_map)}")
    missing_tr = [sid for _, sid, _ in subjects if sid not in tr_map]
    if missing_tr:
        sys.exit(f"no TR for {len(missing_tr)} subjects, e.g. {missing_tr[:10]}. "
                 "Refusing to guess - resolve these or exclude them.")
    print("every subject has a TR: OK")

    # ---- load + resample --------------------------------------------------
    print(f"\nLoading {args.atlas} series, resampling to TR_common="
          f"{args.tr_common}s ...")
    raw_list, res_list, rows, failures = [], [], [], []

    for i, (dirname, sid, label) in enumerate(subjects, 1):
        path = os.path.join(args.data_dir, dirname,
                            f"{dirname}_{args.atlas}_features_timeseries.mat")
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            arr = orient_timeseries(load_mat_array(path), args.n_rois, dirname)
            if not np.isfinite(arr).all():
                raise ValueError("non-finite values in series")
            tr = tr_map[sid]
            res = resample_to_common_grid(arr, tr, args.tr_common)
        except Exception as exc:
            failures.append((dirname, str(exc)))
            continue

        raw_list.append(arr.astype(np.float32))
        res_list.append(res.astype(np.float32))
        rows.append(OrderedDict(
            subject_id=sid,
            folder=dirname,
            label=label,
            tr_native=tr,
            T_native=arr.shape[0],
            duration_sec=round((arr.shape[0] - 1) * tr, 3),
            T_resampled=res.shape[0],
            K_native=native_k(res.shape[0], args.tr_common,
                              args.window_sec, args.stride_sec),
        ))
        if i % 200 == 0:
            print(f"  {i}/{len(subjects)}")

    print(f"\nloaded {len(rows)}, failed {len(failures)}")
    for name, err in failures[:20]:
        print(f"  [FAIL] {name}: {err}")
    if not rows:
        sys.exit("nothing loaded")

    # ---- write HDF5 -------------------------------------------------------
    tag = f"{args.atlas}_tr{args.tr_common:g}"
    h5_path = os.path.join(args.out_dir, f"abide_{tag}_timeseries.h5")

    off_raw = np.cumsum([0] + [a.shape[0] for a in raw_list]).astype(np.int64)
    off_res = np.cumsum([0] + [a.shape[0] for a in res_list]).astype(np.int64)
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ts_raw", data=np.concatenate(raw_list, axis=0),
                         compression="gzip", compression_opts=4)
        f.create_dataset("ts_resampled", data=np.concatenate(res_list, axis=0),
                         compression="gzip", compression_opts=4)
        f.create_dataset("offsets_raw", data=off_raw)
        f.create_dataset("offsets_resampled", data=off_res)
        f.create_dataset("subject_id", data=np.array([r["subject_id"] for r in rows],
                                                     dtype=np.int64))
        f.create_dataset("folder", data=np.array([r["folder"] for r in rows],
                                                 dtype=object), dtype=str_dt)
        f.create_dataset("label", data=np.array([r["label"] for r in rows],
                                                dtype=np.int8))
        f.create_dataset("tr_native", data=np.array([r["tr_native"] for r in rows],
                                                    dtype=np.float64))
        f.attrs.update(dict(
            atlas=args.atlas, n_rois=args.n_rois, tr_common=args.tr_common,
            n_subjects=len(rows), source_dir=os.path.abspath(args.data_dir),
            tr_source=os.path.abspath(args.csv), interp="cubic_spline",
            window_sec_reported=args.window_sec, stride_sec_reported=args.stride_sec,
            git_commit=git_commit(),
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            note=("resampled only - NOT windowed, NOT harmonized, NOT sparsified; "
                  "ComBat and sparsification are fold-dependent and belong in "
                  "the dataloader"),
        ))

    meta_path = os.path.join(args.out_dir, f"abide_{tag}_metadata.csv")
    with open(meta_path, "w", newline="") as fh:
        wr = csvmod.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"wrote {h5_path}  ({os.path.getsize(h5_path)/1e6:.1f} MB)")
    print(f"wrote {meta_path}")
    print("=" * 66)

    labs = Counter(r["label"] for r in rows)
    print(f"\nlabels: control(0)={labs[0]}  patient(1)={labs[1]}")

    print("\nper-TR breakdown (native grid):")
    print(f"  {'TR':>7} {'n':>5} {'T min':>6} {'T max':>6} "
          f"{'dur min':>8} {'dur max':>8}")
    for tr in sorted({r["tr_native"] for r in rows}):
        sub = [r for r in rows if r["tr_native"] == tr]
        Ts = [r["T_native"] for r in sub]
        Ds = [r["duration_sec"] for r in sub]
        print(f"  {tr:7.3f} {len(sub):5d} {min(Ts):6d} {max(Ts):6d} "
              f"{min(Ds):8.1f} {max(Ds):8.1f}")

    Ds = [r["duration_sec"] for r in rows]
    print(f"\nduration overall: min={min(Ds):.1f}s  median="
          f"{np.median(Ds):.1f}s  max={max(Ds):.1f}s")

    kc = Counter(r["K_native"] for r in rows)
    print(f"\nnative K at {args.window_sec:g}s/{args.stride_sec:g}s "
          f"on a {args.tr_common:g}s grid:")
    for k in sorted(kc):
        bar = "#" * min(50, kc[k] * 50 // max(kc.values()))
        print(f"  K={k:3d}  {kc[k]:5d}  {bar}")
    if 0 in kc:
        print(f"  !! {kc[0]} subjects too short for even one window")
    print(f"\n  min K={min(kc)}   max K={max(kc)}   "
          f"-> truncating to a common K would cost the longest scans "
          f"{100*(1 - min(kc)/max(kc)):.0f}% of their windows")


if __name__ == "__main__":
    main()
