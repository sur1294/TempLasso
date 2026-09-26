#!/bin/bash
#SBATCH --job-name=grid_finish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# Finish the four table-row grids after grid_multi: re-drive each grid with grid.sbatch
# (completed tasks return immediately, tasks that failed only the restored-metric guard
# now pass with the documented tolerance, interrupted tasks resume), select-and-test-once,
# then late-fuse the tuned branches for all four blocks.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm
export CUDA_CACHE_PATH=/scratch/ganymede2/$USER/.nv_cache XDG_CACHE_HOME=/scratch/ganymede2/$USER/.cache
ADNI=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5
      TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits)
ABIDE=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5
       TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits)
echo "host=$(hostname) start=$(date '+%F %T')"
env "${ADNI[@]}"  ROW=x         BACKBONE=bqn   EPOCHS=60 PAR=16 bash "$SLURM_DIR/grid.sbatch" > finish_adni_x_bqn.out 2>&1 &
env "${ADNI[@]}"  ROW=x         BACKBONE=alter EPOCHS=60 PAR=4  bash "$SLURM_DIR/grid.sbatch" > finish_adni_x_alter.out 2>&1 &
env "${ADNI[@]}"  ROW=templasso BACKBONE=bqn   EPOCHS=60 PAR=6  bash "$SLURM_DIR/grid.sbatch" > finish_adni_templasso.out 2>&1 &
env "${ABIDE[@]}" ROW=templasso BACKBONE=bqn   EPOCHS=60 PAR=6  bash "$SLURM_DIR/grid.sbatch" > finish_abide_templasso.out 2>&1 &
wait
for f in finish_*.out; do echo "== $f"; tail -n 3 "$f"; done
echo "=== late fusion (tuned branches) $(date '+%F %T')"
bash /scratch/ganymede2/$USER/tbn-runs/grid/late_fusion_tuned.sbatch
echo "end=$(date '+%F %T')"
