#!/bin/bash
#SBATCH --job-name=gcn_abide
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
# ABIDE GCN X (180 tasks) + TempX (240 tasks) on one GPU allocation.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm; S=/scratch/ganymede2/$USER
export CUDA_CACHE_PATH=$S/.nv_cache XDG_CACHE_HOME=$S/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ABIDE=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5 TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits)
sc="${PAR_SCALE:-1}"; par() { python3 -c "import math;print(max(1, math.ceil($1*$sc)))"; }
echo "host=$(hostname) start=$(date '+%F %T')"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
env "${ABIDE[@]}" ROW=x     BACKBONE=gcn EPOCHS=60 PAR=$(par 12) bash "$SLURM_DIR/grid.sbatch" > gcn_abide_x.out 2>&1 &
env "${ABIDE[@]}" ROW=tempx BACKBONE=gcn EPOCHS=60 PAR=$(par 12) bash "$SLURM_DIR/grid.sbatch" > gcn_abide_tempx.out 2>&1 &
wait
for f in gcn_abide_*.out; do echo "== $f"; tail -n 2 "$f"; done; echo "end=$(date '+%F %T')"
