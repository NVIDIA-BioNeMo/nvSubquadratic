#!/bin/bash
#SBATCH --account=all6000users
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=download_well
#SBATCH --output=slurm/download_well_%j.out

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/download_well.sh <dataset_name> [--split train|val|test]"
    exit 1
fi

# Activate conda
source /home/dwessel/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

# Set default data path if not already defined
export WELL_DATA_PATH="${WELL_DATA_PATH:-/ivi/zfs/s0/original_homes/dwessel/data/the_well}"

# Pre-create dataset directories so curl --create-dirs doesn't try to mkdir
# parent paths on the NFS mount (which triggers harmless permission warnings)
DATASET_NAME="$1"
mkdir -p "${WELL_DATA_PATH}/datasets/${DATASET_NAME}/data/train"
mkdir -p "${WELL_DATA_PATH}/datasets/${DATASET_NAME}/data/valid"
mkdir -p "${WELL_DATA_PATH}/datasets/${DATASET_NAME}/data/test"

echo "================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Dataset:      $1"
echo "Destination:  ${WELL_DATA_PATH}"
echo "Started:      $(date)"
echo "================================================"

bash scripts/download_well.sh "$@"

echo "================================================"
echo "Finished: $(date)"
echo "================================================"
