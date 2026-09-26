#!/bin/bash
#SBATCH --job-name=bnt_rows
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#
# BNT rows for one DATASET under OUR protocol (same as the other four backbones):
#   X (bnt grid: layers {1,2} x dropout {.1,.3}) and TempX (x BiGRU {32,64,128} x lr 2) grids,
#   validation-selected per seed, test once; then LATE fusion (probability average, no training)
#   X+TempLasso (TempLasso branch = the existing selected grid) and X+TempX.
# ABIDE/ADNI select on val loss; PPMI on val macro-AUC (grid dirs carry _val_auc_max, matching
# the PPMI re-run). Task dirs are resumable, so --requeue after preemption just continues.
#
#   DS=ABIDE|ADNI|PPMI  sbatch -p gpu-preempt --gres=gpu:nvidia_a100-sxm4-40gb:1 bnt_rows.sbatch
set -uo pipefail
S=/scratch/ganymede2/$USER; SLURM_DIR=$HOME/tbn/experiments/slurm; PY=$HOME/tbn-conda/bin/python
export CUDA_CACHE_PATH=$S/.nv_cache XDG_CACHE_HOME=$S/.cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS="${DS:-ABIDE}"; PAR_X="${PAR_X:-6}"; PAR_T="${PAR_T:-4}"; export DEVICE="${DEVICE:-cuda}"   # DEVICE=cpu for a cpu-preempt copy
case "$DS" in
  ABIDE) E=(DATASET=ABIDE SPLIT_PREFIX=abide DATA_DIR=$HOME/tbn-data STATIC=abide_data_tr.h5 TEMPORAL=derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/splits NUM_CLASSES=2 SELECT_METRIC=val_loss)
         SUF=""; TL=abide_templasso; NC=2; OUT=$HOME/tbn-runs/late_fusion_tuned_v2; ds=abide
         PP=(--dataset ABIDE --split-prefix abide --split-dir $HOME/tbn-data/splits --static-h5 $HOME/tbn-data/abide_data_tr.h5 --temporal-h5 $HOME/tbn-data/derivatives/abide_schaefer100_tr2_w44_s22_temporalfc.h5);;
  ADNI)  E=(DATASET=ADNI SPLIT_PREFIX=adni DATA_DIR=$HOME/tbn-data/adni STATIC=adni_schaefer100_tr2_static_trunc.h5 TEMPORAL=adni_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$HOME/tbn-data/adni/splits NUM_CLASSES=2 SELECT_METRIC=val_loss)
         SUF=""; TL=adni_templasso; NC=2; OUT=$HOME/tbn-runs/late_fusion_tuned_v2; ds=adni
         PP=(--dataset ADNI --split-prefix adni --split-dir $HOME/tbn-data/adni/splits --static-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_static_trunc.h5 --temporal-h5 $HOME/tbn-data/adni/adni_schaefer100_tr2_w44_s22_temporalfc.h5);;
  PPMI)  E=(DATASET=PPMI SPLIT_PREFIX=ppmi DATA_DIR=$S/tbn-data/ppmi STATIC=ppmi_schaefer100_static.h5 TEMPORAL=ppmi_schaefer100_tr2_w44_s22_temporalfc.h5 SPLIT_DIR=$S/tbn-data/ppmi/splits NUM_CLASSES=4 SELECT_METRIC=val_auc)
         SUF="_val_auc_max"; TL=ppmi_templasso_val_auc_max; NC=4; OUT=$HOME/tbn-runs/late_fusion_tuned_ppmi_mauc; ds=ppmi
         PP=(--dataset PPMI --split-prefix ppmi --split-dir $S/tbn-data/ppmi/splits --static-h5 $S/tbn-data/ppmi/ppmi_schaefer100_static.h5 --temporal-h5 $S/tbn-data/ppmi/ppmi_schaefer100_tr2_w44_s22_temporalfc.h5);;
  *) echo "bad DS=$DS"; exit 2;;
esac
echo "host=$(hostname) start=$(date '+%F %T') DS=$DS device=$DEVICE"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
G=$HOME/tbn-runs/grid; cd $G
env "${E[@]}" ROW=x     BACKBONE=bnt EPOCHS=60 PAR=$PAR_X bash "$SLURM_DIR/grid.sbatch" > bnt_${ds}_x.out     2>&1 &
env "${E[@]}" ROW=tempx BACKBONE=bnt EPOCHS=60 PAR=$PAR_T bash "$SLURM_DIR/grid.sbatch" > bnt_${ds}_tempx.out 2>&1 &
wait
for f in bnt_${ds}_x.out bnt_${ds}_tempx.out; do echo "== $f"; tail -n 4 "$f"; done
ok=1; for g in ${ds}_x_bnt${SUF} ${ds}_tempx_bnt${SUF} $TL; do for s in 42 1042 2042 3042 4042; do [ -e "$G/$g/seed$s/selected" ] || { echo "MISSING $g seed$s"; ok=0; }; done; done
[ $ok = 1 ] || { echo "selection incomplete; no fusion"; exit 3; }
echo "=== late fusion $(date '+%F %T')"; export OMP_NUM_THREADS=2; mkdir -p $OUT
LF=$HOME/tbn/experiments/late_fusion_from_checkpoints.py
$PY $LF --repo $HOME/tbn "${PP[@]}" --num-classes $NC --device cpu --x-runs "$G/${ds}_x_bnt${SUF}/seed{seed}/selected" --t-runs "$G/$TL/seed{seed}/selected"              --label bnt_${ds}       --out $OUT/bnt_${ds}.csv       > $OUT/bnt_${ds}.log 2>&1 &
$PY $LF --repo $HOME/tbn "${PP[@]}" --num-classes $NC --device cpu --x-runs "$G/${ds}_x_bnt${SUF}/seed{seed}/selected" --t-runs "$G/${ds}_tempx_bnt${SUF}/seed{seed}/selected" --label bnt_${ds}_tempx --out $OUT/bnt_${ds}_tempx.csv > $OUT/bnt_${ds}_tempx.log 2>&1 &
wait; ls -la $OUT/bnt_${ds}*.csv; echo "end=$(date '+%F %T')"
