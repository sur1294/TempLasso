#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#
# Validation-only grid for one table row, PAR tasks at a time on one GPU node,
# then select-and-test once per seed. Every task is a resumable directory, so
# --requeue after preemption just continues.
#
#   DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5 \
#   TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 \
#   ROW=x BACKBONE=bqn EPOCHS=60 PAR=40 \
#   sbatch -p coskunuzer --job-name=grid_abide_x_bqn grid.sbatch
set -uo pipefail
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=$HOME/tbn-conda/bin/python; REPO=$HOME/tbn; S=$REPO/experiments/run_grid.py
DATASET="${DATASET:-ABIDE}"; SPLIT_PREFIX="${SPLIT_PREFIX:-abide}"; DATA_DIR="${DATA_DIR:-$HOME/tbn-data}"
STATIC="${STATIC:-abide_data_tr.h5}"; TEMPORAL="${TEMPORAL:-derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5}"
SPLIT_DIR="${SPLIT_DIR:-$DATA_DIR/splits}"
ROW="${ROW:-x}"; BACKBONE="${BACKBONE:-bqn}"; EPOCHS="${EPOCHS:-60}"; PAR="${PAR:-40}"
GRID="${GRID:-$HOME/tbn-runs/grid}"
NUM_CLASSES="${NUM_CLASSES:-2}"
DEVICE="${DEVICE:-cuda}"
SELECT_METRIC="${SELECT_METRIC:-val_loss}"
COMMON=(--repo "$REPO" --dataset "$DATASET" --split-prefix "$SPLIT_PREFIX" --split-dir "$SPLIT_DIR"
        --static-h5 "$DATA_DIR/$STATIC" --temporal-h5 "$DATA_DIR/$TEMPORAL" --grid-dir "$GRID"
        --epochs "$EPOCHS" --device "$DEVICE" --row "$ROW" --backbone "$BACKBONE" --num-classes "$NUM_CLASSES" --select-metric "$SELECT_METRIC")
echo "host=$(hostname) start=$(date '+%F %T') $DATASET row=$ROW backbone=$BACKBONE epochs=$EPOCHS par=$PAR num_classes=$NUM_CLASSES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
N=$($PY "$S" list "${COMMON[@]}") || { echo "list failed"; exit 2; }
echo "tasks: $N"
mkdir -p "$GRID/logs"
seq 0 $((N-1)) | xargs -P "$PAR" -I{} sh -c \
  "$PY $S train --task-index {} ${COMMON[*]} > $GRID/logs/${DATASET}_${ROW}_${BACKBONE}_task{}.log 2>&1 || echo 'task {} FAILED'"
echo "training done $(date '+%F %T'); selecting..."
$PY "$S" select "${COMMON[@]}"
echo "end=$(date '+%F %T')"
