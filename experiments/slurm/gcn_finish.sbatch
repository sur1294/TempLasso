#!/bin/bash
#SBATCH --job-name=gcn_finish
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# After the GCN grids: re-drive all six grids on CPU (completed tasks return at once),
# select-and-test-once, then late-fuse GCN X with TempLasso and with GCN TempX.
set -uo pipefail
SLURM_DIR=$HOME/tbn/experiments/slurm; S=/scratch/ganymede2/$USER; export XDG_CACHE_HOME=$S/.cache
ABIDE=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5 TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits NUM_CLASSES=2)
ADNI=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5 TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits NUM_CLASSES=2)
PPMI=(DATASET=PPMI SPLIT_PREFIX=ppmi DATA_DIR=$S/tbn-data/ppmi STATIC=ppmi_schaefer100_static.h5 TEMPORAL=ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$S/tbn-data/ppmi/splits NUM_CLASSES=4)
for row in x tempx; do
  env "${ABIDE[@]}" DEVICE=cpu ROW=$row BACKBONE=gcn EPOCHS=60 PAR=6 bash "$SLURM_DIR/grid.sbatch" > finish_gcn_abide_$row.out 2>&1 &
  env "${ADNI[@]}"  DEVICE=cpu ROW=$row BACKBONE=gcn EPOCHS=60 PAR=4 bash "$SLURM_DIR/grid.sbatch" > finish_gcn_adni_$row.out 2>&1 &
  env "${PPMI[@]}"  DEVICE=cpu ROW=$row BACKBONE=gcn EPOCHS=60 PAR=4 bash "$SLURM_DIR/grid.sbatch" > finish_gcn_ppmi_$row.out 2>&1 &
done
wait
for f in finish_gcn_*.out; do echo "== $f"; tail -n 2 "$f"; done
echo "=== late fusion (GCN) $(date '+%F %T')"; export OMP_NUM_THREADS=2
PY=$HOME/tbn-conda/bin/python; LF=$HOME/tbn/experiments/late_fusion_from_checkpoints.py; G=$HOME/tbn-runs/grid; OUT=$HOME/tbn-runs/late_fusion_tuned
AB=(--dataset ABIDE --split-prefix abide --split-dir $HOME/tbn-data/splits --static-h5 $HOME/tbn-data/abide_data_tr.h5 --temporal-h5 $HOME/tbn-data/derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 --device cpu)
AD=(--dataset ADNI --split-prefix adni --split-dir $HOME/tbn-data/adni/splits --static-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_static_trunc.h5 --temporal-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_w44_s22_temporalfc.h5 --device cpu)
PP=(--dataset PPMI --split-prefix ppmi --split-dir $S/tbn-data/ppmi/splits --static-h5 $S/tbn-data/ppmi/ppmi_schaefer100_static.h5 --temporal-h5 $S/tbn-data/ppmi/ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 --num-classes 4 --device cpu)
run() { local -n F=$1; $PY $LF --repo $HOME/tbn "${F[@]}" --x-runs "$G/$3/seed{seed}/selected" --t-runs "$G/$4/seed{seed}/selected" --label "$2" --out "$OUT/$2.csv"; }
run AB gcn_abide       abide_x_gcn abide_templasso;  run AB gcn_abide_tempx abide_x_gcn abide_tempx_gcn
run AD gcn_adni        adni_x_gcn  adni_templasso;   run AD gcn_adni_tempx  adni_x_gcn  adni_tempx_gcn
run PP gcn_ppmi        ppmi_x_gcn  ppmi_templasso;   run PP gcn_ppmi_tempx  ppmi_x_gcn  ppmi_tempx_gcn
echo "end=$(date '+%F %T')"
