#!/bin/bash
#SBATCH --job-name=grid_race
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.out
# Submit the SAME grid to several GPU types at once; the first copy to start
# takes a lock, cancels its siblings (job ids listed in $RACE_FILE) and runs
# grid.sbatch. Any copy that starts later exits immediately.
set -uo pipefail
GRIDROOT="${GRID:-$HOME/tbn-runs/grid}"; LOCK="$GRIDROOT/race_${RACE_TAG}.lock"
for i in 1 2 3 4 5 6; do [ -s "$RACE_FILE" ] && break; sleep 20; done
if ! mkdir "$LOCK" 2>/dev/null; then
  # A lock left behind by a preempted/killed copy is stale when no sibling is RUNNING.
  ids=$(grep -v "^$SLURM_JOB_ID$" "$RACE_FILE" | tr "\n" "," | sed "s/,$//")
  if [ -n "$ids" ] && [ "$(squeue -h -j "$ids" -t RUNNING 2>/dev/null | wc -l)" != 0 ]; then
    echo "another copy holds $LOCK and is running; exiting"; exit 0
  fi
  echo "stale $LOCK (no running sibling); taking over"; rmdir "$LOCK" 2>/dev/null; mkdir "$LOCK" || exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
for j in $(cat "$RACE_FILE"); do [ "$j" != "$SLURM_JOB_ID" ] && scancel "$j" 2>/dev/null && echo "cancelled sibling $j"; done
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) PAR=${PAR:-} PAR_SCALE=${PAR_SCALE:-}"
bash "${RACE_TARGET:-$HOME/tbn/experiments/slurm/grid.sbatch}"
