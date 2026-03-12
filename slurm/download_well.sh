#!/bin/bash
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=download_well
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/download_well.sh <dataset_name> [--split train|val|test]"
    exit 1
fi

# Activate conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

# Set default data path if not already defined
export WELL_DATA_PATH="${WELL_DATA_PATH:-/gpfs/scratch1/shared/dwessels2/data/the_well}"


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
