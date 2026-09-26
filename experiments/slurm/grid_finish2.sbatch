#!/bin/bash
#SBATCH --job-name=grid_finish2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# After grid_multi2: re-drive the eight grids (completed tasks return at once,
# interrupted ones resume), select-and-test-once, then late-fuse batch 2.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm
export CUDA_CACHE_PATH=/scratch/ganymede2/$USER/.nv_cache XDG_CACHE_HOME=/scratch/ganymede2/$USER/.cache
# 24 GB cards OOM with 8 concurrent TempX-ALTER processes; keep the re-drive narrow.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ADNI=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5
      TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits)
ABIDE=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5
       TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits)
echo "host=$(hostname) start=$(date '+%F %T')"
for bb in bqn alter brainnetcnn; do
  par=4; [ "$bb" = bqn ] && par=6
  env "${ABIDE[@]}" ROW=tempx BACKBONE=$bb EPOCHS=60 PAR=$par bash "$SLURM_DIR/grid.sbatch" > finish2_abide_tempx_$bb.out 2>&1 &
  env "${ADNI[@]}"  ROW=tempx BACKBONE=$bb EPOCHS=60 PAR=3 bash "$SLURM_DIR/grid.sbatch" > finish2_adni_tempx_$bb.out 2>&1 &
done
env "${ABIDE[@]}" ROW=x BACKBONE=brainnetcnn EPOCHS=60 PAR=4 bash "$SLURM_DIR/grid.sbatch" > finish2_abide_x_brainnetcnn.out 2>&1 &
env "${ADNI[@]}"  ROW=x BACKBONE=brainnetcnn EPOCHS=60 PAR=4 bash "$SLURM_DIR/grid.sbatch" > finish2_adni_x_brainnetcnn.out 2>&1 &
wait
for f in finish2_*.out; do echo "== $f"; tail -n 3 "$f"; done
echo "=== late fusion batch 2 $(date '+%F %T')"
bash "$SLURM_DIR/late_fusion_tuned2.sbatch"
echo "end=$(date '+%F %T')"
