#!/bin/bash
#SBATCH --job-name=grid_multi_ppmi
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
# PPMI (4-class, macro OvR AUC) grids on ONE GPU allocation. GROUP=static runs
# X (bqn/alter/brainnetcnn) + TempLasso; GROUP=tempx runs the three TempX rows.
# PAR_SCALE (default 1) shrinks the per-grid parallelism on small cards.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm; S=/scratch/ganymede2/$USER
export CUDA_CACHE_PATH=$S/.nv_cache XDG_CACHE_HOME=$S/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PPMI=(DATASET=PPMI SPLIT_PREFIX=ppmi DATA_DIR=$S/tbn-data/ppmi STATIC=ppmi_schaefer100_static.h5
      TEMPORAL=ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$S/tbn-data/ppmi/splits NUM_CLASSES=4 EPOCHS=60)
sc="${PAR_SCALE:-1}"; par() { python3 -c "import math;print(max(1, math.ceil($1*$sc)))"; }
echo "host=$(hostname) start=$(date '+%F %T') group=$GROUP par_scale=$sc"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [ "$GROUP" = static ]; then
  env "${PPMI[@]}" ROW=x         BACKBONE=bqn         PAR=$(par 12) bash "$SLURM_DIR/grid.sbatch" > ppmi_x_bqn.out 2>&1 &
  env "${PPMI[@]}" ROW=x         BACKBONE=alter       PAR=$(par 4)  bash "$SLURM_DIR/grid.sbatch" > ppmi_x_alter.out 2>&1 &
  env "${PPMI[@]}" ROW=x         BACKBONE=brainnetcnn PAR=$(par 6)  bash "$SLURM_DIR/grid.sbatch" > ppmi_x_brainnetcnn.out 2>&1 &
  env "${PPMI[@]}" ROW=templasso BACKBONE=bqn         PAR=$(par 8)  bash "$SLURM_DIR/grid.sbatch" > ppmi_templasso.out 2>&1 &
else
  env "${PPMI[@]}" ROW=tempx BACKBONE=bqn         PAR=$(par 8) bash "$SLURM_DIR/grid.sbatch" > ppmi_tempx_bqn.out 2>&1 &
  env "${PPMI[@]}" ROW=tempx BACKBONE=alter       PAR=$(par 6) bash "$SLURM_DIR/grid.sbatch" > ppmi_tempx_alter.out 2>&1 &
  env "${PPMI[@]}" ROW=tempx BACKBONE=brainnetcnn PAR=$(par 4) bash "$SLURM_DIR/grid.sbatch" > ppmi_tempx_brainnetcnn.out 2>&1 &
fi
wait
echo "group $GROUP finished end=$(date '+%F %T')"; for f in ppmi_*.out; do echo "== $f"; tail -n 2 "$f"; done
