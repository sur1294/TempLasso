#!/usr/bin/env python3
"""
Step 2 - sliding-window temporal FC from the Step-1 common-grid series
=====================================================================

For every subject:
    (T', R) resampled series
      -> sliding windows (44 s long, 22 s stride, on the TR_common grid)
      -> per-window Ledoit-Wolf shrinkage covariance
      -> normalise covariance to CORRELATION
      -> clip strictly inside (-1, 1)
      -> Fisher-z (arctanh)
      -> store upper triangle (k=1), ragged, float32

Native K throughout: no truncation, no padding.

Deliberately NOT done here (all fold-dependent or ablatable, so they
belong in the dataloader):
    ComBat harmonisation, sparsification, padding/attention masks.

Diagnostics written alongside the data:
    - per-window Ledoit-Wolf shrinkage coefficient lambda
    - per-window count of degenerate (near-zero variance) ROIs
These let you answer "did TR_common bias the estimator by site?" later.

Run the self-test before trusting it:
    python build_step2_temporal_fc.py --self-test
"""

import argparse
import datetime
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required:  pip install h5py")

try:
    from sklearn.covariance import LedoitWolf
except ImportError:
    sys.exit("scikit-learn is required:  pip install scikit-learn")


# clip bound for arctanh; arctanh(1-1e-6) ~= 7.25, comfortable in float32
CLIP = 1.0 - 1e-6
VAR_TOL = 1e-10          # ROI treated as degenerate below this variance


# ---------------------------------------------------------------------------
# core math
# ---------------------------------------------------------------------------

def cov_to_corr(cov):
    """
    Covariance -> correlation.  Degenerate (zero-variance) rows/cols are
    returned as zero off-diagonal with a unit diagonal rather than NaN.
    """
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    bad = d < np.sqrt(VAR_TOL)
    d_safe = np.where(bad, 1.0, d)
    corr = cov / np.outer(d_safe, d_safe)
    if bad.any():
        corr[bad, :] = 0.0
        corr[:, bad] = 0.0
    np.fill_diagonal(corr, 1.0)
    # numerical symmetrisation - guards against 1e-16 asymmetry
    corr = 0.5 * (corr + corr.T)
    return corr, bad


def window_fc(win, standardize=False):
    """
    win : (n_samples, R) one window of the series
    returns (fisher_z_matrix (R,R), shrinkage lambda, n_degenerate_rois)

    DEFAULT (standardize=False) is bit-for-bit identical to the
    supervisor's windowed_fc() in temporal_fc_graphs2.py.  Verified:
    max abs difference 0.0.

    standardize=True z-scores each ROI inside the window first, which
    makes the Ledoit-Wolf shrinkage target the identity ("no
    correlation") rather than a variance-scaled identity.  Arguably
    cleaner when the target quantity is a correlation matrix, but it is
    NOT in the agreed design and it moves edges by ~0.1 in Fisher-z.
    Treat it as an ablation, not a default.
    """
    x = np.asarray(win, dtype=np.float64)
    n, R = x.shape

    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=0)
    degenerate = sd < np.sqrt(VAR_TOL)
    if standardize:
        sd_safe = np.where(degenerate, 1.0, sd)
        x = x / sd_safe
        x[:, degenerate] = 0.0

    lw = LedoitWolf(assume_centered=True).fit(x)
    corr, bad = cov_to_corr(lw.covariance_)

    corr = np.clip(corr, -CLIP, CLIP)
    z = np.arctanh(corr)
    np.fill_diagonal(z, 0.0)          # diagonal carries no information
    return z.astype(np.float64), float(lw.shrinkage_), int(degenerate.sum())


def window_starts(T, w, step):
    """Start indices of every full window; len() == native K."""
    if T < w:
        return []
    return list(range(0, T - w + 1, step))


def upper_tri_idx(R):
    return np.triu_indices(R, k=1)


def vec_to_matrix(vec, R):
    """Rebuild a full symmetric (R,R) matrix from an upper-triangle vector."""
    iu = upper_tri_idx(R)
    m = np.zeros((R, R), dtype=np.float64)
    m[iu] = vec
    m = m + m.T
    return m


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        if not cond:
            ok = False

    print("\n1. cov_to_corr")
    A = rng.normal(size=(200, 8))
    C = np.cov(A, rowvar=False)
    corr, bad = cov_to_corr(C)
    check("diagonal is 1", np.allclose(np.diag(corr), 1.0))
    check("symmetric", np.allclose(corr, corr.T))
    check("matches np.corrcoef", np.allclose(corr, np.corrcoef(A, rowvar=False)),
          f"maxdiff={np.abs(corr-np.corrcoef(A,rowvar=False)).max():.2e}")
    check("no degenerate flagged", not bad.any())

    print("\n2. cov_to_corr with a dead ROI")
    B = A.copy(); B[:, 3] = 0.0
    C2 = np.cov(B, rowvar=False)
    corr2, bad2 = cov_to_corr(C2)
    check("dead ROI flagged", bad2[3] and bad2.sum() == 1)
    check("no NaN/inf", np.isfinite(corr2).all())
    check("dead ROI edges are 0", np.allclose(corr2[3, [0,1,2,4]], 0.0))
    check("dead ROI diagonal is 1", corr2[3, 3] == 1.0)

    print("\n3. Fisher-z round trip")
    r = np.array([-0.9, -0.3, 0.0, 0.45, 0.99])
    check("arctanh/tanh invert", np.allclose(np.tanh(np.arctanh(r)), r))
    check("clip keeps arctanh finite", np.isfinite(np.arctanh(np.clip(1.0, -CLIP, CLIP))))

    print("\n4. Ledoit-Wolf recovers a known correlation structure")
    R = 20
    true = np.eye(R)
    for i in range(R):
        for j in range(i+1, R):
            true[i, j] = true[j, i] = 0.6 ** abs(i-j)      # AR-like decay
    L = np.linalg.cholesky(true)
    big = rng.normal(size=(20000, R)) @ L.T
    z_big, lam_big, _ = window_fc(big)
    est_big = np.tanh(z_big); np.fill_diagonal(est_big, 1.0)
    err_big = np.abs(est_big - true)[~np.eye(R, dtype=bool)].max()
    check("large-n estimate close to truth", err_big < 0.05, f"maxerr={err_big:.4f}")
    check("large-n shrinkage is small", lam_big < 0.05, f"lambda={lam_big:.4f}")

    print("\n5. shrinkage behaves as n shrinks")
    lams, errs_lw, errs_pe = [], [], []
    for n in (2000, 200, 60, 22):
        e_lw, e_pe, ls = [], [], []
        for _ in range(30):
            x = rng.normal(size=(n, R)) @ L.T
            z, lam, _ = window_fc(x)
            m = np.tanh(z); np.fill_diagonal(m, 1.0)
            e_lw.append(np.abs(m - true)[~np.eye(R, dtype=bool)].mean())
            p = np.corrcoef(x, rowvar=False)
            e_pe.append(np.abs(p - true)[~np.eye(R, dtype=bool)].mean())
            ls.append(lam)
        lams.append(np.mean(ls)); errs_lw.append(np.mean(e_lw)); errs_pe.append(np.mean(e_pe))
        print(f"      n={n:5d}  lambda={np.mean(ls):.3f}  "
              f"MAE_LW={np.mean(e_lw):.4f}  MAE_Pearson={np.mean(e_pe):.4f}")
    check("shrinkage grows as n falls", all(a < b for a, b in zip(lams, lams[1:])))
    check("LW beats Pearson at n=22", errs_lw[-1] < errs_pe[-1],
          f"{errs_lw[-1]:.4f} vs {errs_pe[-1]:.4f}")

    print("\n6. output invariants at realistic size (n=22, R=100)")
    x = rng.normal(size=(22, 100))
    z, lam, nd = window_fc(x)
    check("shape", z.shape == (100, 100))
    check("symmetric", np.allclose(z, z.T))
    check("finite", np.isfinite(z).all())
    check("zero diagonal", np.allclose(np.diag(z), 0.0))
    check("lambda in [0,1]", 0.0 <= lam <= 1.0, f"lambda={lam:.4f}")
    ev = np.linalg.eigvalsh(np.tanh(z) + np.eye(100)*(1-0))
    corr_back = np.tanh(z); np.fill_diagonal(corr_back, 1.0)
    check("correlation is positive definite",
          np.linalg.eigvalsh(corr_back).min() > 0,
          f"min eig={np.linalg.eigvalsh(corr_back).min():.2e}")
    check("raw Pearson at n=22 is NOT pos-def (motivates shrinkage)",
          np.linalg.eigvalsh(np.corrcoef(x, rowvar=False)).min() < 1e-8)

    print("\n7. upper-triangle round trip")
    iu = upper_tri_idx(100)
    check("vector length", len(z[iu]) == 4950)
    check("reconstruction exact", np.allclose(vec_to_matrix(z[iu], 100), z))

    print("\n8. window_starts vs Step-1 K formula")
    def k_step1(Tp, w=22, s=11):
        return (Tp - w)//s + 1 if Tp >= w else 0
    bad_k = [(Tp, len(window_starts(Tp,22,11)), k_step1(Tp))
             for Tp in range(22, 320) if len(window_starts(Tp,22,11)) != k_step1(Tp)]
    check("agrees for all T' in [22,320)", not bad_k, f"{bad_k[:5]}")
    check("T'=150 -> K=12", len(window_starts(150,22,11)) == 12)
    check("T'=300 -> K=26", len(window_starts(300,22,11)) == 26)
    check("T'=21 -> K=0", len(window_starts(21,22,11)) == 0)
    check("windows stay in bounds",
          all(s+22 <= 150 for s in window_starts(150,22,11)))

    print("\n9. ragged offsets round trip")
    Ks = [12, 26, 15]
    blocks = [rng.normal(size=(k, 4950)) for k in Ks]
    off = np.cumsum([0]+Ks)
    cat = np.concatenate(blocks, axis=0)
    check("total rows", cat.shape[0] == sum(Ks))
    check("slices recover blocks",
          all(np.allclose(cat[off[i]:off[i+1]], blocks[i]) for i in range(len(Ks))))

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return ok


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the numerical test suite and exit")
    ap.add_argument("--in-h5", help="Step-1 HDF5")
    ap.add_argument("--out-dir")
    ap.add_argument("--window-sec", type=float, default=44.0)
    ap.add_argument("--stride-sec", type=float, default=22.0)
    ap.add_argument("--standardize", action="store_true",
                    help="ABLATION ONLY: z-score ROIs within each window before "
                         "Ledoit-Wolf. Off by default, which reproduces the "
                         "supervisor's windowed_fc() exactly.")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N subjects (debug)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.in_h5 or not args.out_dir:
        ap.error("--in-h5 and --out-dir are required (or use --self-test)")

    os.makedirs(args.out_dir, exist_ok=True)
    standardize = args.standardize

    with h5py.File(args.in_h5, "r") as f:
        tr_common = float(f.attrs["tr_common"])
        R = int(f.attrs["n_rois"])
        atlas = str(f.attrs["atlas"])
        off_in = f["offsets_resampled"][:]
        subject_id = f["subject_id"][:]
        label = f["label"][:]
        tr_native = f["tr_native"][:]
        folder = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in f["folder"][:]]
        n_sub = len(subject_id)
        if args.limit:
            n_sub = min(n_sub, args.limit)

        w = int(round(args.window_sec / tr_common))
        step = int(round(args.stride_sec / tr_common))
        if abs(w*tr_common - args.window_sec) > 1e-9:
            print(f"  NOTE: {args.window_sec}s is not a whole number of "
                  f"{tr_common}s frames; using w={w} frames "
                  f"({w*tr_common}s)")
        if abs(step*tr_common - args.stride_sec) > 1e-9:
            print(f"  NOTE: stride rounded to {step} frames "
                  f"({step*tr_common}s)")
        print(f"window={w} frames ({w*tr_common:g}s)  "
              f"stride={step} frames ({step*tr_common:g}s)  R={R}")
        print(f"standardize within window: {standardize}")

        iu = upper_tri_idx(R)
        n_edges = len(iu[0])
        vecs, lams, degen, Ks = [], [], [], []

        print(f"\nprocessing {n_sub} subjects ...")
        for i in range(n_sub):
            ts = f["ts_resampled"][off_in[i]:off_in[i+1]].astype(np.float64)
            starts = window_starts(ts.shape[0], w, step)
            if not starts:
                raise RuntimeError(
                    f"{folder[i]}: T'={ts.shape[0]} < window {w} - no windows")
            for s in starts:
                z, lam, nd = window_fc(ts[s:s+w], standardize=standardize)
                vecs.append(z[iu].astype(np.float32))
                lams.append(lam)
                degen.append(nd)
            Ks.append(len(starts))
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{n_sub}  ({sum(Ks)} windows)")

    X = np.stack(vecs, axis=0)
    lams = np.asarray(lams, dtype=np.float32)
    degen = np.asarray(degen, dtype=np.int32)
    Ks = np.asarray(Ks, dtype=np.int32)
    off_out = np.concatenate([[0], np.cumsum(Ks)]).astype(np.int64)

    # ---- validation before writing -----------------------------------
    print("\nvalidating ...")
    assert X.shape == (Ks.sum(), n_edges), X.shape
    assert off_out[-1] == X.shape[0]
    assert np.isfinite(X).all(), "non-finite edges produced"
    assert (lams >= 0).all() and (lams <= 1).all()
    print(f"  {X.shape[0]} windows x {n_edges} edges, all finite: OK")

    tag = f"{atlas}_tr{tr_common:g}_w{args.window_sec:g}_s{args.stride_sec:g}"
    out = os.path.join(args.out_dir, f"abide_{tag}_temporalfc.h5")
    with h5py.File(out, "w") as g:
        g.create_dataset("fc_z_upper", data=X, compression="gzip",
                         compression_opts=4, chunks=(min(256, X.shape[0]), n_edges))
        g.create_dataset("offsets", data=off_out)
        g.create_dataset("K", data=Ks)
        g.create_dataset("shrinkage", data=lams)
        g.create_dataset("n_degenerate_rois", data=degen)
        g.create_dataset("subject_id", data=subject_id[:n_sub])
        g.create_dataset("label", data=label[:n_sub])
        g.create_dataset("tr_native", data=tr_native[:n_sub])
        g.create_dataset("triu_row", data=iu[0].astype(np.int16))
        g.create_dataset("triu_col", data=iu[1].astype(np.int16))
        g.attrs.update(dict(
            atlas=atlas, n_rois=R, n_edges=n_edges, tr_common=tr_common,
            window_sec=args.window_sec, stride_sec=args.stride_sec,
            window_frames=w, stride_frames=step,
            estimator="LedoitWolf->correlation->clip->arctanh",
            standardized_within_window=standardize,
            clip=CLIP, storage="upper triangle k=1, diagonal excluded",
            source=os.path.abspath(args.in_h5),
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            note=("NOT harmonized, NOT sparsified, NOT padded; ComBat and "
                  "sparsification are fold-dependent -> dataloader"),
        ))

    # ---- report -------------------------------------------------------
    print("\n" + "="*66)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("="*66)
    print(f"\nsubjects {len(Ks)}   windows {X.shape[0]}   edges/window {n_edges}")
    print(f"K: min={Ks.min()} max={Ks.max()} mean={Ks.mean():.1f}")
    print(f"\nshrinkage lambda: min={lams.min():.3f} median={np.median(lams):.3f} "
          f"max={lams.max():.3f}")
    print("  lambda by native TR (site-effect diagnostic):")
    per_sub_lam = np.array([lams[off_out[i]:off_out[i+1]].mean()
                            for i in range(len(Ks))])
    for tr in sorted(set(tr_native[:n_sub].tolist())):
        m = tr_native[:n_sub] == tr
        print(f"    TR={tr:5.3f}  n={m.sum():4d}  mean lambda={per_sub_lam[m].mean():.4f}")
    if degen.sum():
        print(f"\n  !! {int((degen>0).sum())} windows had degenerate ROIs "
              f"(max {int(degen.max())} in one window)")
    else:
        print("\n  no degenerate ROIs: OK")

    print(f"\nedge values (Fisher-z): mean={X.mean():.4f} sd={X.std():.4f} "
          f"min={X.min():.3f} max={X.max():.3f}")
    from scipy.stats import skew
    sk = skew(X.ravel()[::997])
    print(f"  skewness (subsampled) = {sk:.4f}"
          f"   {'-> near-symmetric, Fisher-z fine' if abs(sk) < 0.5 else '-> consider Box-Cox'}")

    print("\nread back with:")
    print("  off = g['offsets'][:]; v = g['fc_z_upper'][off[i]:off[i+1]]  # (K, 4950)")


if __name__ == "__main__":
    main()
