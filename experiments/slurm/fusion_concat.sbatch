#!/bin/bash
#SBATCH --job-name=fusion_concat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#
# X+TempX and X+TempLasso as CONCAT JOINT TRAINING (the paper's "simple fusion") for one DATASET,
# a list of backbones and a list of fusion rows, on ONE GPU allocation. Everything else is our
# protocol: same frozen splits, 60 epochs, ABIDE/ADNI val-loss selection, PPMI val macro-AUC,
# test once per seed (run_grid.py select). Task dirs are resumable; --requeue continues.
#
#   DS=ABIDE ROWS="fusion_tempx fusion_templasso" BACKBONES="bqn alter" PAR_TX=8 PAR_TL=10 \
#     sbatch -p coskunuzer fusion_concat.sbatch
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm; S=/scratch/ganymede2/$USER
export CUDA_CACHE_PATH=$S/.nv_cache XDG_CACHE_HOME=$S/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS="${DS:-ABIDE}"; ROWS="${ROWS:-fusion_tempx fusion_templasso}"; BACKBONES="${BACKBONES:-bqn}"
PAR_TX="${PAR_TX:-8}"; PAR_TL="${PAR_TL:-10}"; DEVICE="${DEVICE:-cuda}"
case "$DS" in
  ABIDE) E=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5 TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits NUM_CLASSES=2 SELECT_METRIC=val_loss);;
  ADNI)  E=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5 TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits NUM_CLASSES=2 SELECT_METRIC=val_loss);;
  PPMI)  E=(DATASET=PPMI SPLIT_PREFIX=ppmi DATA_DIR=$S/tbn-data/ppmi STATIC=ppmi_schaefer100_static.h5 TEMPORAL=ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$S/tbn-data/ppmi/splits NUM_CLASSES=4 SELECT_METRIC=val_auc);;
  *) echo "bad DS=$DS"; exit 2;;
esac
echo "host=$(hostname) start=$(date '+%F %T') DS=$DS rows=[$ROWS] backbones=[$BACKBONES] device=$DEVICE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd $HOME/tbn-runs/grid
for row in $ROWS; do for bb in $BACKBONES; do
  par=$PAR_TL; [ "$row" = fusion_tempx ] && par=$PAR_TX
  env "${E[@]}" ROW=$row BACKBONE=$bb EPOCHS=60 PAR=$par DEVICE=$DEVICE bash "$SLURM_DIR/grid.sbatch" > concat_${DS}_${row}_${bb}.out 2>&1 &
done; done
wait
for row in $ROWS; do for bb in $BACKBONES; do echo "== concat_${DS}_${row}_${bb}.out"; tail -n 3 concat_${DS}_${row}_${bb}.out; done; done
echo "end=$(date '+%F %T')"
