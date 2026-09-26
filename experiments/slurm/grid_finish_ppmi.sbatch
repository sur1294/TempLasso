#!/bin/bash
#SBATCH --job-name=grid_finish_ppmi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# After the PPMI races: re-drive the seven grids (completed tasks return at once,
# interrupted ones resume), select-and-test-once, then 4-class late fusion.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm; S=/scratch/ganymede2/$USER
export CUDA_CACHE_PATH=$S/.nv_cache XDG_CACHE_HOME=$S/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PPMI=(DATASET=PPMI SPLIT_PREFIX=ppmi DATA_DIR=$S/tbn-data/ppmi STATIC=ppmi_schaefer100_static.h5
      TEMPORAL=ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$S/tbn-data/ppmi/splits NUM_CLASSES=4 EPOCHS=60)
echo "host=$(hostname) start=$(date '+%F %T')"
for bb in bqn alter brainnetcnn; do
  env "${PPMI[@]}" ROW=x     BACKBONE=$bb PAR=4 bash "$SLURM_DIR/grid.sbatch" > finish_ppmi_x_$bb.out 2>&1 &
  env "${PPMI[@]}" ROW=tempx BACKBONE=$bb PAR=3 bash "$SLURM_DIR/grid.sbatch" > finish_ppmi_tempx_$bb.out 2>&1 &
done
env "${PPMI[@]}" ROW=templasso BACKBONE=bqn PAR=4 bash "$SLURM_DIR/grid.sbatch" > finish_ppmi_templasso.out 2>&1 &
wait
for f in finish_ppmi_*.out; do echo "== $f"; tail -n 2 "$f"; done
echo "=== late fusion (4-class) $(date '+%F %T')"
export OMP_NUM_THREADS=2
PY=$HOME/tbn-conda/bin/python; LF=$HOME/tbn/experiments/late_fusion_from_checkpoints.py; G=$HOME/tbn-runs/grid; OUT=$HOME/tbn-runs/late_fusion_tuned; mkdir -p $OUT
D=(--dataset PPMI --split-prefix ppmi --split-dir $S/tbn-data/ppmi/splits --static-h5 $S/tbn-data/ppmi/ppmi_schaefer100_static.h5
   --temporal-h5 $S/tbn-data/ppmi/ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 --num-classes 4)
for bb in bqn alter brainnetcnn; do
  $PY $LF --repo $HOME/tbn "${D[@]}" --x-runs "$G/ppmi_x_$bb/seed{seed}/selected" --t-runs "$G/ppmi_templasso/seed{seed}/selected" --label ${bb}_ppmi       --out $OUT/${bb}_ppmi.csv
  $PY $LF --repo $HOME/tbn "${D[@]}" --x-runs "$G/ppmi_x_$bb/seed{seed}/selected" --t-runs "$G/ppmi_tempx_$bb/seed{seed}/selected" --label ${bb}_ppmi_tempx --out $OUT/${bb}_ppmi_tempx.csv
done
echo "end=$(date '+%F %T')"
