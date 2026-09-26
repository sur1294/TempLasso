#!/bin/bash
#SBATCH --job-name=tempx_sgrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#
# TempX for one DATASET/BACKBONE with the collaborator's temporal search space (run_grid --tempx-grid),
# our protocol otherwise (60 epochs, val-loss selection, test once), then X+TempX late fusion against the
# existing selected X grid. Same mechanics as grid.sbatch (resumable task dirs; --requeue continues).
#   DS=ABIDE BACKBONE=bqn TEMPX_GRID=surbhi_l1 PAR=10 sbatch -p coskunuzer tempx_sgrid.sbatch
set -uo pipefail
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=/scratch/ganymede2/$USER; export CUDA_CACHE_PATH=$S/.nv_cache XDG_CACHE_HOME=$S/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/tbn-conda/bin/python; REPO=$HOME/tbn; RG=$REPO/experiments/run_grid.py; LF=$REPO/experiments/late_fusion_from_checkpoints.py
DS="${DS:-ABIDE}"; BACKBONE="${BACKBONE:-bqn}"; TEMPX_GRID="${TEMPX_GRID:-surbhi_l1}"; PAR="${PAR:-10}"; EPOCHS="${EPOCHS:-60}"; DEVICE="${DEVICE:-cuda}"
case "$DS" in
  ABIDE) PP=(--dataset ABIDE --split-prefix abide --split-dir $HOME/tbn-data/splits --static-h5 $HOME/tbn-data/abide_data_tr.h5 --temporal-h5 $HOME/tbn-data/derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 --num-classes 2); SEL=val_loss; SUF=""; OUT=$HOME/tbn-runs/late_fusion_tuned_v2; ds=abide;;
  ADNI)  PP=(--dataset ADNI --split-prefix adni --split-dir $HOME/tbn-data/adni/splits --static-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_static_trunc.h5 --temporal-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_w44_s22_temporalfc.h5 --num-classes 2); SEL=val_loss; SUF=""; OUT=$HOME/tbn-runs/late_fusion_tuned_v2; ds=adni;;
  PPMI)  PP=(--dataset PPMI --split-prefix ppmi --split-dir $S/tbn-data/ppmi/splits --static-h5 $S/tbn-data/ppmi/ppmi_schaefer100_static.h5 --temporal-h5 $S/tbn-data/ppmi/ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 --num-classes 4); SEL=val_auc; SUF="_val_auc_max"; OUT=$HOME/tbn-runs/late_fusion_tuned_ppmi_mauc; ds=ppmi;;
  *) echo "bad DS"; exit 2;;
esac
G=$HOME/tbn-runs/grid; TAG=$([ "$TEMPX_GRID" = surbhi ] && echo _sgrid || echo _sgrid1)
COMMON=(--repo $REPO "${PP[@]}" --grid-dir $G --epochs $EPOCHS --device $DEVICE --row tempx --backbone $BACKBONE --select-metric $SEL --tempx-grid $TEMPX_GRID)
echo "host=$(hostname) start=$(date '+%F %T') $DS tempx $BACKBONE grid=$TEMPX_GRID par=$PAR"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
N=$($PY $RG list "${COMMON[@]}") || { echo "list failed"; exit 2; }; echo "tasks: $N"; mkdir -p $G/logs
seq 0 $((N-1)) | xargs -P "$PAR" -I{} sh -c "$PY $RG train --task-index {} ${COMMON[*]} > $G/logs/${DS}_tempx_${BACKBONE}${TAG}_task{}.log 2>&1 || echo 'task {} FAILED'"
echo "training done $(date '+%F %T'); selecting..."; $PY $RG select "${COMMON[@]}" || { echo "select failed"; exit 3; }
echo "=== X+TempX late fusion $(date '+%F %T')"; export OMP_NUM_THREADS=2; mkdir -p $OUT
$PY $LF --repo $REPO "${PP[@]}" --device cpu --x-runs "$G/${ds}_x_${BACKBONE}${SUF}/seed{seed}/selected" --t-runs "$G/${ds}_tempx_${BACKBONE}${TAG}${SUF}/seed{seed}/selected" --label ${BACKBONE}_${ds}_tempx${TAG} --out $OUT/${BACKBONE}_${ds}_tempx${TAG}.csv
echo "end=$(date '+%F %T')"
