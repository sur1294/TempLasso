#!/bin/bash
#SBATCH --job-name=lf_all_v2
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
# Recompute every tuned late fusion (4 backbones x 3 datasets x {TempLasso, TempX}) with the
# extended metrics (AUC, ACC, SEN, SPE). Inference only; branches are the frozen selected winners.
set -u; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME=/scratch/ganymede2/$USER/.cache
S=/scratch/ganymede2/$USER; PY=$HOME/tbn-conda/bin/python; LF=$HOME/tbn/experiments/late_fusion_from_checkpoints.py; G=$HOME/tbn-runs/grid; OUT=$HOME/tbn-runs/late_fusion_tuned_v2; mkdir -p $OUT
AB=(--dataset ABIDE --split-prefix abide --split-dir $HOME/tbn-data/splits --static-h5 $HOME/tbn-data/abide_data_tr.h5 --temporal-h5 $HOME/tbn-data/derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 --device cpu)
AD=(--dataset ADNI --split-prefix adni --split-dir $HOME/tbn-data/adni/splits --static-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_static_trunc.h5 --temporal-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_w44_s22_temporalfc.h5 --device cpu)
PP=(--dataset PPMI --split-prefix ppmi --split-dir $S/tbn-data/ppmi/splits --static-h5 $S/tbn-data/ppmi/ppmi_schaefer100_static.h5 --temporal-h5 $S/tbn-data/ppmi/ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 --num-classes 4 --device cpu)
run() { local -n F=$1; $PY $LF --repo $HOME/tbn "${F[@]}" --x-runs "$G/$3/seed{seed}/selected" --t-runs "$G/$4/seed{seed}/selected" --label "$2" --out "$OUT/$2.csv" > $OUT/$2.log 2>&1 && echo "done $2" || echo "FAILED $2"; }
for bb in bqn alter brainnetcnn gcn; do
  run AB ${bb}_abide abide_x_$bb abide_templasso & run AB ${bb}_abide_tempx abide_x_$bb abide_tempx_$bb &
  run AD ${bb}_adni  adni_x_$bb  adni_templasso  & run AD ${bb}_adni_tempx  adni_x_$bb  adni_tempx_$bb  &
  run PP ${bb}_ppmi  ppmi_x_$bb  ppmi_templasso  & run PP ${bb}_ppmi_tempx  ppmi_x_$bb  ppmi_tempx_$bb  &
  wait
done
echo "lf_all_v2 done $(date '+%F %T')"
