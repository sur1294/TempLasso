# TempLasso — experiments, data preparation, and tests

Code for *Representation over Capacity: Sparse Temporal Access for Functional Connectivity*.
`unified_bn/` (framework, backbones, TempLasso selector, BiGRU, pipelines) and `models/`
(reference backbone implementations) are in the repository root; this branch adds the
scripts that produced the reported tables.

## Layout

```
experiments/run_grid.py                   one (dataset, row, backbone) hyper-parameter grid:
                                          train every candidate (resumable), select per seed on the
                                          validation split, evaluate the winner on test exactly once
experiments/late_fusion_from_checkpoints.py
                                          X+TempLasso / X+TempX as prediction-level fusion of the two
                                          selected branches (probability average, no training)
experiments/build_result_tables.py        result CSVs -> per-seed TSV / mean±std tables
experiments/slurm/                        the sbatch drivers used on the cluster
data_prep/abide/                          ROI time series -> sliding-window Ledoit-Wolf FC (Fisher-z) -> splits
data_prep/adni/build_adni_artifacts.py    ADNI static/temporal artefacts (TR 2 s, 288 s truncation)
data_prep/ppmi/build_static_from_collab_h5.py
data_prep/make_dataset_splits.py          stratified 70/10/20 subject splits (ADNI: label x TR group)
data_prep/splits/{abide,adni,ppmi}/       the frozen splits (subject ids + labels) for seeds 42, 1042, 2042, 3042, 4042
tests/                                    pytest suite (python -m pytest -q tests)
```

## Protocol

* Rows: `x` (static backbone), `tempx` (backbone applied per window + BiGRU), `templasso`
  (train-only L1 edge selection on pooled training windows + BiGRU), and the concat
  joint-training variants `fusion_tempx` / `fusion_templasso`.
* Backbones: `bqn`, `alter`, `brainnetcnn`, `gcn`, `bnt`.
* 60 epochs, Adam (weight decay 1e-4, cosine to 1e-5), batch 16, no early stopping.
  Per seed, the checkpoint (epoch) and the candidate are selected on the validation split
  (`--select-metric val_loss`, or `val_auc` = macro one-vs-rest AUC for PPMI with `--num-classes 4`);
  the test split is evaluated once for the selected candidate.
* Splits in `data_prep/splits/` are frozen and shared by every method; do not regenerate them.

## Running one grid

```bash
COMMON=(--repo . --dataset ABIDE --row templasso --backbone bqn --split-prefix abide
        --split-dir data_prep/splits/abide --static-h5 /path/abide_data_tr.h5
        --temporal-h5 /path/abide_schaefer100_tr2_w44_s22_temporalfc.h5
        --grid-dir runs/grid --epochs 60 --device cuda)          # PPMI: add --num-classes 4 --select-metric val_auc
python experiments/run_grid.py list  "${COMMON[@]}"              # number of (candidate, seed) tasks
python experiments/run_grid.py train "${COMMON[@]}" --task-index 0   # 0..N-1, parallelisable, resumable
python experiments/run_grid.py select "${COMMON[@]}" --out selected.csv
python experiments/late_fusion_from_checkpoints.py --repo . --dataset ABIDE --split-prefix abide \
  --split-dir data_prep/splits/abide --static-h5 ... --temporal-h5 ... --label bqn_abide --out lf.csv \
  --x-runs 'runs/grid/abide_x_bqn/seed{seed}/selected' --t-runs 'runs/grid/abide_templasso/seed{seed}/selected'
```

Data artefacts (HDF5) and run directories are not part of the repository.
