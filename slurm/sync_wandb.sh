#!/bin/bash
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=backfill,batch
#SBATCH --mail-type=FAIL
#SBATCH --job-name=healthcareeng_research-nvsubq.wbsync

# Usage:
#   sbatch slurm/sync_wandb.sh <run_dir> [<run_dir> ...]
#
# Each <run_dir> is an absolute path (or repo-relative) to a Lightning RESULTS dir
# that contains a wandb/ subfolder with offline-run-*-* sessions to sync to W&B.
#
# Submits one slurm job that syncs every <run_dir> sequentially. To parallelize
# across many run dirs, use slurm/sync_wandb_submit_all.sh which launches one
# slurm job per run dir.

set -u

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch slurm/sync_wandb.sh <run_dir> [<run_dir> ...]"
    exit 1
fi

WANDB_BIN="${WANDB_BIN:-/home/amoradzadeh/.local/bin/wandb}"
HYENA_ROOT="/lustre/fsw/healthcareeng_bionemo/amoradzadeh/hyena"

LOG_DIR="${HYENA_ROOT}/vit_film/slurm/wandb_sync_logs"
mkdir -p "${LOG_DIR}"

echo "================================================"
echo "W&B sync job ${SLURM_JOB_ID} on $(hostname)"
echo "Started: $(date)"
echo "Run dirs to sync: $#"
echo "wandb bin: ${WANDB_BIN}"
echo "================================================"

for arg in "$@"; do
    # Allow both absolute and repo-relative paths
    if [[ "$arg" = /* ]]; then
        d="$arg"
    else
        d="${HYENA_ROOT}/${arg}"
    fi

    echo ""
    echo "================================================"
    echo "[$(date +%H:%M:%S)] STARTING: $d"
    echo "================================================"

    if [ ! -d "$d/wandb" ]; then
        echo "[$(date +%H:%M:%S)] SKIP: $d/wandb does not exist"
        continue
    fi

    cd "$d"
    for run_dir in wandb/run-*-* wandb/offline-run-*-*; do
        [ -d "$run_dir" ] || continue
        echo "[$(date +%H:%M:%S)] sync $run_dir"
        "${WANDB_BIN}" sync "$run_dir"
        rc=$?
        echo "[$(date +%H:%M:%S)] done $run_dir (rc=${rc})"
    done

    echo "[$(date +%H:%M:%S)] FINISHED: $d"
done

echo ""
echo "[$(date +%H:%M:%S)] ALL SYNCS COMPLETE"
