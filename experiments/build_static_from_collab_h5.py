#!/usr/bin/env python3
"""
Dense static-FC artefact from a collaborator per-subject HDF5
(``<subject>/<atlas>/{corr_matrix,timeseries}`` + attrs TR_s/label/age/sex),
in the layout the unified pipeline's StaticABIDEDataset reads:
``subject_ids, labels, fc [N,R,R], tr_native, tr_group, age, sex``.

Subjects are taken in lexically sorted group order (the same order Step 2
used for the temporal file). Labels are preserved as delivered (PPMI is
4-class: 0 Control, 1 PD, 2 SWEDD, 3 Prodromal). ``fc`` is the delivered
``corr_matrix`` (whole scan), i.e. the same convention as ABIDE's
``abide_data_tr.h5``; the label-confound-driven 288 s truncation used for
ADNI is not applied here.
"""
import argparse, datetime, os, subprocess
import h5py, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in-h5", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--atlas", default="schaefer100"); ap.add_argument("--dataset", required=True)
ap.add_argument("--slow-tr-threshold", type=float, default=None,
                help="tr_group = 'slow' if TR_s >= threshold else 'fast' (default: median split)")
a = ap.parse_args()
ids, labels, fc, tr, age, sex = [], [], [], [], [], []
with h5py.File(a.in_h5) as f:
    for k in sorted(f.keys()):
        g = f[k]; c = g[f"{a.atlas}/corr_matrix"][:]
        assert c.ndim == 2 and c.shape[0] == c.shape[1] and np.isfinite(c).all(), k
        ids.append(k); labels.append(int(g.attrs["label"])); fc.append(c.astype(np.float32))
        tr.append(float(g.attrs["TR_s"])); age.append(float(g.attrs["age"])); sex.append(int(g.attrs["sex"]))
tr = np.asarray(tr); thr = a.slow_tr_threshold if a.slow_tr_threshold is not None else float(np.median(tr))
tr_group = np.where(tr >= thr, "slow", "fast").astype("S")
labels = np.asarray(labels, dtype=np.int64); fc = np.stack(fc)
try: commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
except Exception: commit = "unknown"
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with h5py.File(a.out, "w") as o:
    o.create_dataset("subject_ids", data=np.asarray(ids, dtype="S")); o.create_dataset("labels", data=labels)
    o.create_dataset("fc", data=fc, compression="gzip"); o.create_dataset("tr_native", data=tr)
    o.create_dataset("tr_group", data=tr_group); o.create_dataset("age", data=np.asarray(age)); o.create_dataset("sex", data=np.asarray(sex))
    o.attrs.update(dict(dataset=a.dataset, atlas=a.atlas, n_rois=fc.shape[1], source=os.path.abspath(a.in_h5),
                        fc_estimator="collaborator corr_matrix, whole scan, as delivered", git_commit=commit,
                        created=datetime.datetime.now().isoformat(timespec="seconds"), slow_tr_threshold=thr,
                        label_counts=str(dict(zip(*map(lambda x: x.tolist(), np.unique(labels, return_counts=True)))))))
print(f"wrote {a.out}: N={len(ids)} fc={fc.shape} labels={dict(zip(*map(lambda x: x.tolist(), np.unique(labels, return_counts=True))))} tr_groups={dict(zip(*map(lambda x: x.tolist(), np.unique(tr_group, return_counts=True))))} (thr {thr})")
