#!/bin/bash
# Submit a linear chain of Well training jobs to work around the 4-hour wall-time limit.
# Each job starts after the previous one finishes (regardless of exit code),
# and autoresume picks up the latest checkpoint automatically.
#
# Usage (from repo root):
#   bash slurm/queue_well.sh <num_jobs> <config.py> [config_overrides...]
#
# Examples:
#   bash slurm/queue_well.sh 6 examples/well/v2/active_matter/hyena_gaussian_mask.py
#   bash slurm/queue_well.sh 6 examples/well/v2/active_matter/hyena_gaussian_mask.py net.in_proj_cfg.patch_size=8
#   bash slurm/queue_well.sh 6 examples/well/v2/active_matter/hyena_gaussian_mask.py net.in_proj_cfg.patch_size=8 start_after_jid=5039096

if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_jobs> <config.py> [config_overrides...] [start_after_jid=<jid>]"
    exit 1
fi

SCRIPT_NAME="slurm/submit_well.sh"
NUM_JOBS="$1"
CONFIG="$2"
shift 2

START_AFTER=""
CONFIG_OVERRIDES=()
for arg in "$@"; do
    if [[ "${arg}" == start_after_jid=* ]]; then
        START_AFTER="${arg#start_after_jid=}"
    else
        CONFIG_OVERRIDES+=("${arg}")
    fi
done

submit_job() {
    local dep_flag="${1:-}"
    sbatch ${dep_flag} "${SCRIPT_NAME}" "${CONFIG}" \
        "${CONFIG_OVERRIDES[@]+"${CONFIG_OVERRIDES[@]}"}" \
        | awk '{print $4}'
}

if [ -n "${START_AFTER}" ]; then
    jid_prev=$(submit_job "--dependency=afterany:${START_AFTER}")
    echo "Submitted job 1/${NUM_JOBS}: ${jid_prev}  (after ${START_AFTER})"
else
    jid_prev=$(submit_job)
    echo "Submitted job 1/${NUM_JOBS}: ${jid_prev}"
fi

for i in $(seq 2 "${NUM_JOBS}"); do
    jid_prev=$(submit_job "--dependency=afterany:${jid_prev}")
    echo "Submitted job ${i}/${NUM_JOBS}: ${jid_prev}"
done

echo ""
echo "Chain submitted. Last job ID: ${jid_prev}"
echo "Monitor with: squeue -u oviessmann"
