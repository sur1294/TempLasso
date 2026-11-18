#!/usr/bin/env bash
#SBATCH --job-name=templasso_grid
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Submit as an array; obtain the size with run_grid.py list}"
: "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
cd -- "$SLURM_SUBMIT_DIR"
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -f experiments/run_grid.py ]]; then
    echo "Run sbatch from the repository root." >&2
    exit 1
fi

exec "$PYTHON_BIN" -u experiments/run_grid.py train \
    --task-index "$SLURM_ARRAY_TASK_ID" "$@"
