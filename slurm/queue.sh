#!/bin/bash
if [ $# -ge 2 ]; then
    SCRIPT_NAME="$1"
    NUM_JOBS="$2"
else
    echo "Usage: $0 <script_name> <num_jobs>"
    exit 1
fi

jid_prev=$(sbatch ${SCRIPT_NAME} | awk '{print $4}')

for i in $(seq 2 ${NUM_JOBS}); do
    jid_prev=$(sbatch --dependency=afterany:${jid_prev} ${SCRIPT_NAME} | awk '{print $4}')
    echo "Submitted chained job $i with id $jid_prev"
done
