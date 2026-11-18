# TempLasso

Code for static functional-connectivity predictors, temporal predictors, and
static–temporal fusion on ABIDE, ADNI, and PPMI.

TempLasso selects connections with L1-regularized logistic regression fitted
on training windows. It standardizes their values using training statistics,
then passes the ordered sequence to a bidirectional GRU and classification
head. TempX applies a learned backbone to each FC window before the BiGRU.

This release contains the main workflow and synthetic verification tests.
Datasets, subject split files, trained weights, experiment results, and
ablation experiments are not included.

## Contents

| Location | Purpose |
| --- | --- |
| `models/` | BQN, ALTER, BrainNetCNN, BNT, and GCN implementations |
| `unified_bn/` | Data loading, L1 selection, temporal aggregation, fusion, training, and metrics |
| `experiments/run_grid.py` | Candidate training, validation selection, and selected-model test evaluation |
| `experiments/grid_protocol.py` | Grid completeness checks, frozen selection, and result collection |
| `experiments/late_fusion_from_checkpoints.py` | Fusion of independently selected models |
| `build_step1_timeseries.py` | ABIDE ROI time-series loading and common-TR resampling |
| `build_step2_temporal_fc.py` | Windowed shrinkage FC and Fisher-z edge artifacts |
| `build_adni_artifacts.py` | ADNI artifact construction from a hierarchical input HDF5 |
| `experiments/build_static_from_collab_h5.py` | Conversion of existing hierarchical static FC to dense HDF5 |
| `make_dataset_splits.py` | Split creation for a new study; existing studies should reuse their split files |
| `scripts/train_grid.sbatch` | Portable Slurm array worker |
| `tests/` | Synthetic tests of models, data contracts, training, and selection |

The backbone keys are `bqn`, `alter`, `brainnetcnn`, `bnt`, and `gcn`.
Implementation details and upstream references are in the corresponding
`models/` documentation. In particular, the `bnt` key identifies the supplied
GraphTransformer implementation described in `models/bnt/UPSTREAM.md`.

## Dependencies

Use an existing Python environment with PyTorch, NumPy, SciPy, h5py, and
scikit-learn. The supported baseline is Python 3.9 or newer. Install missing
dependencies as needed:

```bash
python -m pip install -r requirements.txt
```

For GPU training, use a PyTorch build compatible with the available CUDA
runtime. CPU execution is supported with `--device cpu`. No script creates
or activates an environment automatically. Run commands from this directory.

## Data and frozen splits

Training accepts prepared HDF5 files and presaved subject-level splits.
Although several loader classes have `ABIDE` in their names, the shared
runtime also supports ADNI and four-class PPMI.

The canonical static HDF5 contains:

| Dataset | Shape / meaning |
| --- | --- |
| `subject_ids` | `[N]`, unique subject identifiers |
| `labels` | `[N]`, integer class labels |
| `fc` | `[N, R, R]`, whole-scan correlation matrices |

The canonical temporal HDF5 contains:

| Dataset | Shape / meaning |
| --- | --- |
| `subject_id`, `label` | `[N]`, identifiers and labels |
| `fc_z_upper` | `[W, E]`, Fisher-z upper-triangle edges for all windows |
| `K` | `[N]`, number of windows per subject |
| `offsets` | `[N+1]`, cumulative window offsets |
| `triu_row`, `triu_col` | `[E]`, indices in `numpy.triu_indices(R, k=1)` order |

For 100 ROIs, `E = 4,950`. The loader reconstructs correlation matrices
from the stored Fisher-z values. TempLasso extracts and standardizes the
selected Fisher-z edges inside the model; do not pre-standardize its input
with validation or test statistics.

Each split file is named `<prefix>_split_seed<seed>.npz` and contains
`train_ids`, `val_ids`, and `test_ids`. The default seeds are 42, 1042,
2042, 3042, and 4042. The subject partitions must be disjoint within each
split, and static and temporal files must agree on subject IDs and labels.
The loaders preserve the requested split order.

| Dataset | `--dataset` | `--num-classes` | Checkpoint and candidate selection | Test metric |
| --- | --- | --- | --- | --- |
| ABIDE | `ABIDE` | `2` | Minimum validation loss | ROC-AUC |
| ADNI | `ADNI` | `2` | Minimum validation loss | ROC-AUC |
| PPMI | `PPMI` | `4` | Maximum validation macro-AUC | Macro one-vs-rest ROC-AUC |

Labels must be encoded consistently as `0..num_classes-1`. Use the original
class mapping, prepared artifacts, and frozen splits when repeating an
existing experiment. This code-only package does not supply those artifacts.

## Main training workflow

These Bash examples use paths that you supply. Replace each `/path/to/...`
with the corresponding local file or directory. Set a new output directory
for a new experiment.

```bash
REPO="$PWD"
DATASET=ABIDE
PREFIX=abide
N_CLASSES=2
STATIC_H5=/path/to/data/abide_static.h5
TEMPORAL_H5=/path/to/data/abide_temporal_fc.h5
SPLITS=/path/to/frozen_splits
GRID_DIR="$REPO/runs/main_grid"
PYTHON=python

COMMON=(
  --repo "$REPO" --dataset "$DATASET" --num-classes "$N_CLASSES"
  --split-prefix "$PREFIX" --split-dir "$SPLITS"
  --static-h5 "$STATIC_H5" --temporal-h5 "$TEMPORAL_H5"
  --grid-dir "$GRID_DIR" --expected-n-rois 100
  --epochs 200 --batch-size 16 --device cuda
)
MODEL=(--row templasso)

# Inspect the number of candidate/seed tasks; this does not train.
N_TASKS=$("$PYTHON" experiments/run_grid.py list "${COMMON[@]}" "${MODEL[@]}")
echo "$N_TASKS"

# Train one task. Use an allocated compute node for GPU work.
"$PYTHON" experiments/run_grid.py train "${COMMON[@]}" "${MODEL[@]}" --task-index 0
```

The 200-epoch setting above is explicit; the runner's historical default
is 60. Keep all arguments identical between training, collection, freezing,
and testing. For ADNI, change the dataset, prefix, and input paths. For PPMI,
also set `N_CLASSES=4` before constructing `COMMON`.

TempLasso has 18 candidates per seed:

| Dataset | L1 logistic-regression `C` | BiGRU width per direction | Learning rate |
| --- | --- | --- | --- |
| ABIDE | 0.005, 0.01, 0.03 | 32, 64, 128 | 0.0001, 0.0003 |
| ADNI / PPMI | 0.01, 0.03, 0.1 | 32, 64, 128 | 0.0001, 0.0003 |

Tasks are indexed in seed-major, candidate-minor order: 90 tasks for the
default five-seed TempLasso grid. `C` is the inverse regularization strength.
Training uses Adam, cosine learning-rate decay, and validation-based
checkpoint selection. The complete serialized configuration is saved with
each run. All candidate training is validation-only.

To run another main model, replace `MODEL` and repeat the workflow:

```bash
MODEL=(--row x --backbone brainnetcnn)                 # static X
MODEL=(--row tempx --backbone brainnetcnn)             # window encoder + BiGRU
MODEL=(--row fusion --backbone brainnetcnn)            # joint X + TempX
MODEL=(--row fusion_templasso --backbone brainnetcnn)  # joint X + TempLasso
```

Use one assignment at a time. Each backbone has its own predefined grid;
run `list` to obtain its task count. The joint fusion rows concatenate
branch embeddings and train a classification head jointly. They are
distinct from probability averaging of independently trained models.

### Slurm

Submit from the repository root using the same `COMMON` and `MODEL` arrays.
Use a partition available on the local cluster:

```bash
mkdir -p logs
export PYTHON
sbatch --partition=YOUR_GPU_PARTITION --array="0-$((N_TASKS-1))" \
  scripts/train_grid.sbatch "${COMMON[@]}" "${MODEL[@]}"
```

The template requests one GPU, eight CPUs, 32 GB RAM, and 24 hours per
task. Override these at submission if required. No account, partition,
email, environment path, or array concurrency limit is hardcoded. If
`python` is not the intended interpreter, set `PYTHON` to its absolute path.

### Collect, freeze, and test

After **all** tasks complete:

```bash
"$PYTHON" experiments/run_grid.py collect "${COMMON[@]}" "${MODEL[@]}"
"$PYTHON" experiments/run_grid.py freeze  "${COMMON[@]}" "${MODEL[@]}"
"$PYTHON" experiments/run_grid.py select  "${COMMON[@]}" "${MODEL[@]}"
```

`collect` reads saved metrics without evaluating models. `freeze` requires
a complete grid and records each seed's validation winner and checkpoint
hash. `select` verifies the frozen winners before testing them. A repeated
`select` reuses recorded test metrics. Test scores do not select candidates.
Do not train further candidates inside a grid after its winners are frozen.

Each grid is saved below:

```text
runs/main_grid/<dataset>_<row>[_<backbone>]/
  seed42/cand_<index>_<fingerprint>/candidate_<index>_<fingerprint>/
    best.pt
    last.pt
    history.json
    result.json
  frozen_validation_winners.json
  selected_rows.csv
  csv/all_hyperparameters.csv
  csv/seed42.csv
  ...
```

Exact candidate suffixes are generated by the runner. `last.pt` supports
resuming an interrupted task; `best.pt` is selected on validation.
Non-selected candidates retain validation results and checkpoints; their
test metrics are not produced by the normal grid workflow. CSV AUC values
are fractions in `[0,1]`. Printed grid and late-fusion summaries use
percentages and sample standard deviation across seeds (`ddof=1`); with
only one seed the standard deviation is undefined and printed as `nan`.

## Late fusion of independently trained models

First finish validation selection for both branches. On Linux, `select`
creates a `seed<seed>/selected` symlink to each winning candidate search
directory. The runner also saves `selected.json` with explicit paths.

```bash
"$PYTHON" experiments/late_fusion_from_checkpoints.py \
  --repo "$REPO" --dataset "$DATASET" --num-classes "$N_CLASSES" \
  --split-prefix "$PREFIX" --split-dir "$SPLITS" \
  --static-h5 "$STATIC_H5" --temporal-h5 "$TEMPORAL_H5" \
  --x-runs "$GRID_DIR/abide_x_brainnetcnn/seed{seed}/selected" \
  --t-runs "$GRID_DIR/abide_templasso/seed{seed}/selected" \
  --label brainnetcnn_abide --device cpu \
  --out "$REPO/runs/late_fusion_brainnetcnn_abide.csv"
```

Adjust both run patterns and the output name for another cohort/backbone.
On systems without the convenience symlink, provide the candidate search
directories referenced by `selected.json`, evaluating one seed at a time
with `--seeds` if needed.

The `avg` row is the fixed probability average `0.5 * (p_X + p_T)`. The
script also reports other late-fusion rules; any fitted mixing weights or
stacking model use validation data only. It does not automatically compare
these rules against jointly trained concatenation models or choose a
reported strategy using test results. Choose the analysis protocol before
inspecting test results.

## Preprocessing entry points

Use existing prepared artifacts when reproducing a run. For new inputs,
the following entry points describe and validate their expected formats:

```bash
python build_step1_timeseries.py --help
python build_step2_temporal_fc.py --help
python build_adni_artifacts.py --help
python experiments/build_static_from_collab_h5.py --help
python make_dataset_splits.py --help
```

The ABIDE time-series builder reads per-subject ROI MAT files and a metadata
CSV containing TR. The temporal builder reads the resampled HDF5 and uses
44-second windows, a 22-second stride, Ledoit–Wolf covariance, conversion
to correlation, and Fisher-z upper-triangle storage by default. It does
not perform supervised edge selection; that happens inside training.

The ADNI builder accepts hierarchical subject/atlas HDF5 inputs. It writes
both the supplied whole-scan FC and a separately recomputed FC artifact;
choose the artifact specified by the experiment protocol. The static
converter preserves supplied FC values and labels. PPMI training accepts
prepared artifacts in the shared schemas above; this release does not
include a raw PPMI acquisition-to-ROI preprocessing pipeline.

For a new study, `make_dataset_splits.py` can create stratified 70/10/20
train/validation/test splits. It is not part of training or evaluation and
must not be used to replace an existing study's frozen partitions.

## Verification

Tests construct synthetic data in temporary directories; they do not need
the clinical datasets or trained checkpoints.

```bash
python -m pip install -r requirements-dev.txt
python build_step2_temporal_fc.py --self-test
python -m pytest -q
```

The suite checks TRAIN-only selector fitting, data alignment, multiclass
metrics, backbone/temporal/fusion interfaces, checkpointing, and a small
end-to-end train/freeze/test workflow. Passing these tests verifies software
behavior; reproducing a reported experiment also requires its original
inputs, splits, configurations, and software environment.
