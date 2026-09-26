#!/bin/bash
#SBATCH --job-name=late_fusion_tuned
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
# Late fusion of the *tuned* X / TempLasso branches: each seed's branch is the
# validation-selected winner of its grid (grid/<ds>_<row>/seed<seed>/selected).
# No training; the held-out test is read once per fusion rule.
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
$PY $S --repo $HOME/tbn "${AB[@]}" --x-runs "$G/abide_x_bqn/seed{seed}/selected"   --t-runs "$G/abide_templasso/seed{seed}/selected" --label bqn_abide_tuned   --out $OUT/bqn_abide.csv
$PY $S --repo $HOME/tbn "${AB[@]}" --x-runs "$G/abide_x_alter/seed{seed}/selected" --t-runs "$G/abide_templasso/seed{seed}/selected" --label alter_abide_tuned --out $OUT/alter_abide.csv
$PY $S --repo $HOME/tbn "${AD[@]}" --x-runs "$G/adni_x_bqn/seed{seed}/selected"    --t-runs "$G/adni_templasso/seed{seed}/selected"  --label bqn_adni_tuned    --out $OUT/bqn_adni.csv
$PY $S --repo $HOME/tbn "${AD[@]}" --x-runs "$G/adni_x_alter/seed{seed}/selected"  --t-runs "$G/adni_templasso/seed{seed}/selected"  --label alter_adni_tuned  --out $OUT/alter_adni.csv
echo "late_fusion_tuned done"
