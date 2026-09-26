#!/bin/bash
#SBATCH --job-name=tbqn_bigru_3042
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=22:00:00
#SBATCH --partition=turing
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
TEMPORAL_H5="${PROJECT}/derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5"
SPLIT_FILE="${PROJECT}/splits/abide_split_seed3042.npz"
RESULTS_DIR="${PROJECT}/results/temporal_bqn_bigru_tuning"

SEED=3042

# Frozen BQN hyperparameters from the completed static-BQN validation selection.
BQN_LAYERS=4
BQN_ACTIVATION="elu"
BQN_DROPOUT=0.1

# First-stage temporal grid only:
# 2 hidden sizes x 2 GRU depths x 4 dropouts = 16 runs / seed.
GRU_HIDDEN=(32 64)
GRU_LAYERS=(1 2)
GRU_DROPOUT=(0.0 0.1 0.2 0.3)

N_H=${#GRU_HIDDEN[@]}
N_L=${#GRU_LAYERS[@]}
N_D=${#GRU_DROPOUT[@]}
TOTAL=$(( N_H * N_L * N_D ))

IDX="${SLURM_ARRAY_TASK_ID}"

if (( IDX < 0 || IDX >= TOTAL )); then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX}, expected 0..$((TOTAL-1))"
    exit 1
fi

# Mixed-radix decoding:
# fastest-changing: dropout
# then GRU layers
# slowest-changing: hidden size
DROP_IDX=$(( IDX % N_D ))
TMP=$(( IDX / N_D ))

LAYER_IDX=$(( TMP % N_L ))
HIDDEN_IDX=$(( TMP / N_L ))

HIDDEN="${GRU_HIDDEN[$HIDDEN_IDX]}"
GRU_L="${GRU_LAYERS[$LAYER_IDX]}"
GRU_DROP="${GRU_DROPOUT[$DROP_IDX]}"

# Fixed first-pass temporal settings.
ATT_HIDDEN=32
CLF_HIDDEN=64
CLF_DROP=0.1

LN_TAG="ln1"

RUN_DIR="${RESULTS_DIR}/seed${SEED}_bqnL${BQN_LAYERS}_${BQN_ACTIVATION}_bqndrop${BQN_DROPOUT}_gruH${HIDDEN}_gruL${GRU_L}_grudrop${GRU_DROP}_att${ATT_HIDDEN}_clfH${CLF_HIDDEN}_clfdrop${CLF_DROP}_${LN_TAG}"
RESULT_JSON="${RUN_DIR}/result.json"

echo "================================================================================"
echo "[TEMPORAL BQN + BiGRU GRID]"
echo "seed        : ${SEED}"
echo "array task  : ${IDX}/$((TOTAL-1))"
echo "BQN         : L=${BQN_LAYERS} act=${BQN_ACTIVATION} drop=${BQN_DROPOUT}"
echo "BiGRU       : H=${HIDDEN} L=${GRU_L} drop=${GRU_DROP}"
echo "attention   : ${ATT_HIDDEN}"
echo "classifier  : H=${CLF_HIDDEN} drop=${CLF_DROP} LayerNorm=True"
echo "split       : ${SPLIT_FILE}"
echo "result dir  : ${RUN_DIR}"
echo "host        : $(hostname)"
echo "start       : $(date)"
echo "================================================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

if [[ ! -f "${TEMPORAL_H5}" ]]; then
    echo "[ERROR] Missing temporal HDF5: ${TEMPORAL_H5}"
    exit 1
fi

if [[ ! -f "${SPLIT_FILE}" ]]; then
    echo "[ERROR] Missing split file: ${SPLIT_FILE}"
    exit 1
fi

# Safe resume/requeue behavior: a finished validation-only run has result.json.
if [[ -f "${RESULT_JSON}" ]]; then
    echo "[SKIP] Completed result already exists: ${RESULT_JSON}"
    exit 0
fi

conda run -n alter python models/bqn/bqn_plus/train_temporal_bqn_bigru.py \
    --temporal-h5 "${TEMPORAL_H5}" \
    --split "${SPLIT_FILE}" \
    --seed "${SEED}" \
    --bqn-layers "${BQN_LAYERS}" \
    --bqn-dropout "${BQN_DROPOUT}" \
    --bqn-activation "${BQN_ACTIVATION}" \
    --cluster-num 4 \
    --bigru-hidden-dim "${HIDDEN}" \
    --bigru-num-layers "${GRU_L}" \
    --bigru-dropout "${GRU_DROP}" \
    --attention-hidden-dim "${ATT_HIDDEN}" \
    --classifier-hidden-dim "${CLF_HIDDEN}" \
    --classifier-dropout "${CLF_DROP}" \
    --temporal-layer-norm \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --lr-min 1e-5 \
    --lr-schedule cosine \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --label-smoothing 0.0 \
    --num-workers 4 \
    --device cuda \
    --out-dir "${RESULTS_DIR}"

echo "================================================================================"
echo "[DONE]"
echo "seed=${SEED} H=${HIDDEN} L=${GRU_L} drop=${GRU_DROP}"
echo "finished: $(date)"
echo "================================================================================"
