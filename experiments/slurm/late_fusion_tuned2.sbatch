#!/bin/bash
#SBATCH --job-name=late_fusion_tuned2
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
# Late fusion of tuned branches, second batch: X + TempX for the three backbones
# and X + TempLasso for BrainNetCNN, on ABIDE and ADNI. No training.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=/scratch/ganymede2/$USER/.cache
PY=$HOME/tbn-conda/bin/python; S=$HOME/tbn/experiments/late_fusion_from_checkpoints.py
G=$HOME/tbn-runs/grid; OUT=$HOME/tbn-runs/late_fusion_tuned; mkdir -p "$OUT"
AB=(--dataset ABIDE --split-prefix abide --split-dir $HOME/tbn-data/splits
    --static-h5 $HOME/tbn-data/abide_data_tr.h5
    --temporal-h5 $HOME/tbn-data/derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5)
AD=(--dataset ADNI --split-prefix adni --split-dir $HOME/tbn-data/adni/splits
    --static-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_static_trunc.h5
    --temporal-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_w44_s22_temporalfc.h5)
run() { # ds_flags_name label x_grid t_grid
  local -n F=$1; $PY $S --repo $HOME/tbn "${F[@]}" --x-runs "$G/$3/seed{seed}/selected" --t-runs "$G/$4/seed{seed}/selected" --label "$2" --out "$OUT/$2.csv"; }
for bb in bqn alter brainnetcnn; do
  run AB ${bb}_abide_tempx abide_x_$bb abide_tempx_$bb
  run AD ${bb}_adni_tempx  adni_x_$bb  adni_tempx_$bb
done
run AB brainnetcnn_abide abide_x_brainnetcnn abide_templasso
run AD brainnetcnn_adni  adni_x_brainnetcnn  adni_templasso
echo "late_fusion_tuned2 done"
