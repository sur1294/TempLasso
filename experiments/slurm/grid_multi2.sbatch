#!/bin/bash
#SBATCH --job-name=grid_multi2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=56
#SBATCH --mem=220G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
# Second batch of table-row grids on ONE GPU allocation: BrainNetCNN X and the
# three TempX rows, on ABIDE and ADNI. Same mechanics as grid_multi.sbatch.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm
export CUDA_CACHE_PATH=/scratch/ganymede2/$USER/.nv_cache XDG_CACHE_HOME=/scratch/ganymede2/$USER/.cache
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
ADNI=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5
      TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits)
ABIDE=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5
       TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits)
echo "host=$(hostname) start=$(date '+%F %T')"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
env "${ABIDE[@]}" ROW=x     BACKBONE=brainnetcnn EPOCHS=60 PAR=6  bash "$SLURM_DIR/grid.sbatch" > multi2_abide_x_brainnetcnn.out 2>&1 &
env "${ADNI[@]}"  ROW=x     BACKBONE=brainnetcnn EPOCHS=60 PAR=5  bash "$SLURM_DIR/grid.sbatch" > multi2_adni_x_brainnetcnn.out 2>&1 &
env "${ABIDE[@]}" ROW=tempx BACKBONE=bqn         EPOCHS=60 PAR=10 bash "$SLURM_DIR/grid.sbatch" > multi2_abide_tempx_bqn.out 2>&1 &
env "${ABIDE[@]}" ROW=tempx BACKBONE=alter       EPOCHS=60 PAR=8  bash "$SLURM_DIR/grid.sbatch" > multi2_abide_tempx_alter.out 2>&1 &
env "${ABIDE[@]}" ROW=tempx BACKBONE=brainnetcnn EPOCHS=60 PAR=6  bash "$SLURM_DIR/grid.sbatch" > multi2_abide_tempx_brainnetcnn.out 2>&1 &
env "${ADNI[@]}"  ROW=tempx BACKBONE=bqn         EPOCHS=60 PAR=8  bash "$SLURM_DIR/grid.sbatch" > multi2_adni_tempx_bqn.out 2>&1 &
env "${ADNI[@]}"  ROW=tempx BACKBONE=alter       EPOCHS=60 PAR=5  bash "$SLURM_DIR/grid.sbatch" > multi2_adni_tempx_alter.out 2>&1 &
env "${ADNI[@]}"  ROW=tempx BACKBONE=brainnetcnn EPOCHS=60 PAR=4  bash "$SLURM_DIR/grid.sbatch" > multi2_adni_tempx_brainnetcnn.out 2>&1 &
wait
echo "all grids finished end=$(date '+%F %T')"
for f in multi2_*.out; do echo "== $f"; tail -n 3 "$f"; done
