#!/bin/bash
#SBATCH --job-name=bqn_schaefer_42
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-59
#SBATCH --time=22:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --exclude=g-03-06
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --chdir=/groups/bcoskunuzer/sxk230046/fall2026/Brainnetwork
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sxk230046@utdallas.edu

set -euo pipefail
mkdir -p logs

PROJECT="/groups/bcoskunuzer/sxk230046/fall2026/Brainnetwork"
STATIC_H5="${PROJECT}/derivatives/abide_data_tr.h5"
SPLIT_FILE="${PROJECT}/splits/abide_split_seed42.npz"
RESULTS_DIR="${PROJECT}/results/bqn_tuning"

SEED=42

LAYERS=(1 2 3 4 5)
DROPOUTS=(0.0 0.1 0.2 0.3)
ACTIVATIONS=(gelu leaky_relu elu)

N_LAYERS=${#LAYERS[@]}
N_DROPOUTS=${#DROPOUTS[@]}
N_ACTS=${#ACTIVATIONS[@]}
TOTAL=$(( N_LAYERS * N_DROPOUTS * N_ACTS ))

IDX="${SLURM_ARRAY_TASK_ID}"

if (( IDX < 0 || IDX >= TOTAL )); then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX}, expected 0..$((TOTAL-1))"
    exit 1
fi

ACT_IDX=$(( IDX % N_ACTS ))
TMP=$(( IDX / N_ACTS ))

DROPOUT_IDX=$(( TMP % N_DROPOUTS ))
TMP=$(( TMP / N_DROPOUTS ))

LAYER_IDX=$(( TMP % N_LAYERS ))

LAYERS_N="${LAYERS[$LAYER_IDX]}"
DROPOUT="${DROPOUTS[$DROPOUT_IDX]}"
ACTIVATION="${ACTIVATIONS[$ACT_IDX]}"

RUN_DIR="${RESULTS_DIR}/seed${SEED}_L${LAYERS_N}_${ACTIVATION}_drop${DROPOUT}"
RESULT_JSON="${RUN_DIR}/result.json"

echo "======================================================================"
echo "[BQN SCHAEFER100 GRID]"
echo "seed       : ${SEED}"
echo "array task : ${IDX}/$((TOTAL-1))"
echo "layers     : ${LAYERS_N}"
echo "dropout    : ${DROPOUT}"
echo "activation : ${ACTIVATION}"
echo "split      : ${SPLIT_FILE}"
echo "result dir : ${RUN_DIR}"
echo "host       : $(hostname)"
echo "start      : $(date)"
echo "======================================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

if [[ ! -f "${STATIC_H5}" ]]; then
    echo "[ERROR] Missing static HDF5: ${STATIC_H5}"
    exit 1
fi

if [[ ! -f "${SPLIT_FILE}" ]]; then
    echo "[ERROR] Missing split file: ${SPLIT_FILE}"
    exit 1
fi

if [[ -f "${RESULT_JSON}" ]]; then
    echo "[SKIP] Completed result already exists: ${RESULT_JSON}"
    exit 0
fi

conda run -n alter python models/bqn/train_bqn.py \
    --h5 "${STATIC_H5}" \
    --split "${SPLIT_FILE}" \
    --seed "${SEED}" \
    --layers "${LAYERS_N}" \
    --dropout "${DROPOUT}" \
    --activation "${ACTIVATION}" \
    --cluster-num 4 \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --lr-min 1e-5 \
    --weight-decay 1e-4 \
    --lr-schedule cosine \
    --num-workers 4 \
    --device cuda \
    --out-dir "${RESULTS_DIR}"

echo "======================================================================"
echo "[DONE]"
echo "seed=${SEED} layers=${LAYERS_N} dropout=${DROPOUT} activation=${ACTIVATION}"
echo "finished: $(date)"
echo "======================================================================"
