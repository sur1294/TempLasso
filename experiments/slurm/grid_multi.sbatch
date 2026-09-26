#!/bin/bash
#SBATCH --job-name=grid_multi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#
# Run several table-row grids concurrently on ONE GPU allocation (small models,
# CPU-bound tasks; the GPU is shared). Each grid is grid.sbatch run as a plain
# script with its own env, PAR budget and log. Tasks are resumable directories,
# so --requeue after preemption just continues.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm
export CUDA_CACHE_PATH=/scratch/ganymede2/$USER/.nv_cache XDG_CACHE_HOME=/scratch/ganymede2/$USER/.cache
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
ADNI=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5
      TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits)
ABIDE=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5
       TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits)
echo "host=$(hostname) start=$(date '+%F %T')"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
env "${ADNI[@]}"  ROW=x         BACKBONE=bqn   EPOCHS=60 PAR=16 bash "$SLURM_DIR/grid.sbatch" > multi_adni_x_bqn.out 2>&1 &
env "${ADNI[@]}"  ROW=templasso BACKBONE=bqn   EPOCHS=60 PAR=8  bash "$SLURM_DIR/grid.sbatch" > multi_adni_templasso.out 2>&1 &
env "${ADNI[@]}"  ROW=x         BACKBONE=alter EPOCHS=60 PAR=4  bash "$SLURM_DIR/grid.sbatch" > multi_adni_x_alter.out 2>&1 &
env "${ABIDE[@]}" ROW=templasso BACKBONE=bqn   EPOCHS=60 PAR=16 bash "$SLURM_DIR/grid.sbatch" > multi_abide_templasso.out 2>&1 &
wait
echo "all grids finished end=$(date '+%F %T')"
for f in multi_*.out; do echo "== $f"; tail -n 4 "$f"; done
