#!/bin/bash
if [ $# -ge 1 ]; then
    NUM_JOBS="$1"
else
    NUM_JOBS=10
fi


jid_prev=$(sbatch submit.sh | awk '{print $4}')

for i in $(seq 2 ${NUM_JOBS}); do
    jid_prev=$(sbatch --dependency=afterany:${jid_prev} submit.sh | awk '{print $4}')
    echo "Submitted chained job $i with id $jid_prev"
done
