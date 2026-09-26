#!/bin/bash
#SBATCH --job-name=gcn_cpu
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
# GCN X + TempX grids for one small cohort (ADNI or PPMI) on CPU, as the collaborator
# suggested. Each grid = grid.sbatch with DEVICE=cpu; tasks are resumable dirs.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm; S=/scratch/ganymede2/$USER
export XDG_CACHE_HOME=$S/.cache
DS="${DS:?adni|ppmi}"
if [ "$DS" = adni ]; then
  ENVV=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5 TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits NUM_CLASSES=2)
else
  ENVV=(DATASET=PPMI SPLIT_PREFIX=ppmi DATA_DIR=$S/tbn-data/ppmi STATIC=ppmi_schaefer100_static.h5 TEMPORAL=ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$S/tbn-data/ppmi/splits NUM_CLASSES=4)
fi
echo "host=$(hostname) start=$(date '+%F %T') ds=$DS cpus=$SLURM_CPUS_PER_TASK"
env "${ENVV[@]}" DEVICE=cpu ROW=x     BACKBONE=gcn EPOCHS=60 PAR=20 bash "$SLURM_DIR/grid.sbatch" > gcn_cpu_${DS}_x.out 2>&1 &
env "${ENVV[@]}" DEVICE=cpu ROW=tempx BACKBONE=gcn EPOCHS=60 PAR=24 bash "$SLURM_DIR/grid.sbatch" > gcn_cpu_${DS}_tempx.out 2>&1 &
wait
for f in gcn_cpu_${DS}_*.out; do echo "== $f"; tail -n 2 "$f"; done; echo "end=$(date '+%F %T')"
