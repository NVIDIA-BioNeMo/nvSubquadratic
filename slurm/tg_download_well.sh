#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=download_well
#SBATCH --output=slurm/download_well_%j.out

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: sbatch slurm/tg_download_well.sh <data_path> <dataset_name> [--split train|val|test]"
    exit 1
fi

# Activate conda
source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

export WELL_DATA_PATH="$1"
DATASET_NAME="$2"
shift 2

# Pre-create dataset directories so curl --create-dirs doesn't try to mkdir
# parent paths on the NFS mount (which triggers harmless permission warnings)
mkdir -p "${WELL_DATA_PATH}/datasets/${DATASET_NAME}/data/train"
mkdir -p "${WELL_DATA_PATH}/datasets/${DATASET_NAME}/data/valid"
mkdir -p "${WELL_DATA_PATH}/datasets/${DATASET_NAME}/data/test"

echo "================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "Dataset:      ${DATASET_NAME}"
echo "Destination:  ${WELL_DATA_PATH}"
echo "Started:      $(date)"
echo "================================================"

bash scripts/download_well.sh "${DATASET_NAME}" "$@"

echo "================================================"
echo "Finished: $(date)"
echo "================================================"
