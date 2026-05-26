#!/bin/bash
# Submit a linear chain of jobs to work around the 4-hour wall-time limit.
# Each job starts after the previous one finishes (regardless of exit code),
# and autoresume picks up the latest checkpoint automatically.
#
# Usage:
#   bash queue.sh <submit_script.sh> <num_jobs> [start_after_jid]
#
# Examples:
#   # Chain 10 jobs for hyena_patch16 (fresh start)
#   bash slurm/queue.sh slurm/submit_in1k_cls.sh 10 \
#       examples/vit5_imagenet/v5_patch/hyena_patch16.py
#
#   # Continue an existing chain after job 4434820
#   bash slurm/queue.sh slurm/submit_in1k_cls.sh 5 \
#       examples/vit5_imagenet/v5_patch/hyena_patch16.py start_after_jid=4434820

if [ $# -lt 3 ]; then
    echo "Usage: $0 <submit_script.sh> <num_jobs> <config.py> [extra_sbatch_args...] [start_after_jid=<jid>]"
    echo ""
    echo "Examples:"
    echo "  bash $0 slurm/submit_in1k_cls.sh 10 examples/vit5_imagenet/v5_patch/hyena_patch16.py"
    echo "  bash $0 slurm/submit_in1k_cls.sh 5  examples/vit5_imagenet/v5_patch/hyena_patch8.py start_after_jid=4434820"
    exit 1
fi

SCRIPT_NAME="$1"
NUM_JOBS="$2"
CONFIG="$3"
shift 3

# Parse optional start_after_jid=<id> from remaining args; collect the rest as sbatch args
START_AFTER=""
SBATCH_EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == start_after_jid=* ]]; then
        START_AFTER="${arg#start_after_jid=}"
    else
        SBATCH_EXTRA_ARGS+=("${arg}")
    fi
done

submit_job() {
    local dep_flag="${1:-}"
    sbatch ${dep_flag} "${SCRIPT_NAME}" "${CONFIG}" "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}" \
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
echo "Monitor with: squeue -j ${jid_prev}"
