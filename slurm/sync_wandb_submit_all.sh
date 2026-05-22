#!/bin/bash
# Host-side helper: discover every run dir under vit_film/runs and vit5_multinode/runs
# that contains offline W&B sessions, then submit one CPU-only slurm job per run dir
# via slurm/sync_wandb.sh. Jobs run in parallel.
#
# Usage:
#   bash slurm/sync_wandb_submit_all.sh                # submit all run dirs with offline sessions
#   bash slurm/sync_wandb_submit_all.sh --dry-run      # just list what would be submitted
#   bash slurm/sync_wandb_submit_all.sh <run_dir>...   # submit only the named run dirs (skips offline-session check)

set -u

HYENA_ROOT="/lustre/fsw/healthcareeng_bionemo/amoradzadeh/hyena"
SBATCH_SCRIPT="${HYENA_ROOT}/vit_film/slurm/sync_wandb.sh"
LOG_DIR="${HYENA_ROOT}/vit_film/slurm/wandb_sync_logs"
mkdir -p "${LOG_DIR}"

DRY_RUN=0
EXPLICIT_DIRS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        *) EXPLICIT_DIRS+=("$arg") ;;
    esac
done

if [ "${#EXPLICIT_DIRS[@]}" -gt 0 ]; then
    DIRS=("${EXPLICIT_DIRS[@]}")
else
    # Auto-discover every run_* dir with at least one offline-run session
    mapfile -t DIRS < <(
        find "${HYENA_ROOT}/vit_film/runs" "${HYENA_ROOT}/vit5_multinode/runs" \
            -maxdepth 2 -type d -name "run_*" 2>/dev/null |
        while read -r d; do
            n=$(ls -d "$d"/wandb/offline-run-*-* 2>/dev/null | wc -l)
            if [ "$n" -gt 0 ]; then
                echo "$d"
            fi
        done | sort
    )
fi

echo "Found ${#DIRS[@]} run dirs to sync"
echo ""

submitted=0
for d in "${DIRS[@]}"; do
    if [[ "$d" != /* ]]; then
        d="${HYENA_ROOT}/${d}"
    fi
    n_off=$(ls -d "$d"/wandb/offline-run-*-* 2>/dev/null | wc -l)
    label=$(basename "$(dirname "$d")")/$(basename "$d")
    short_log=$(echo "$label" | tr '/' '_')
    out="${LOG_DIR}/sync-${short_log}-%j.out"

    cmd=(sbatch
        --output="${out}"
        --error="${out}"
        --job-name="healthcareeng_research-nvsubq.wbsync.${short_log}"
        "${SBATCH_SCRIPT}"
        "$d")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] off=${n_off}  ${label}"
        echo "             cmd: ${cmd[*]}"
    else
        echo "  submit off=${n_off}  ${label}"
        "${cmd[@]}"
        submitted=$((submitted+1))
    fi
done

if [ "$DRY_RUN" -eq 0 ]; then
    echo ""
    echo "Submitted ${submitted} sync jobs. Logs under ${LOG_DIR}/"
    echo "Monitor with: squeue -u \$USER -n wbsync.<name>  or  squeue -u \$USER | grep wbsync"
fi
