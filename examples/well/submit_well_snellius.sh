#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_a100
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch [--job-name=NAME] examples/well/submit_well.sh <config.py> [extra args...]"
    echo "  e.g. sbatch --job-name=well-hyena examples/well/submit_well.sh examples/well/active_matter/cfg_hyena.py"
    exit 1
fi

CONFIG="$1"
shift

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export WELL_DATA_PATH="${WELL_DATA_PATH:-/gpfs/scratch1/shared/dwessels2/data/the_well/datasets}"

# CUDA module
module load 2025
module load CUDA/12.8.0

# NCCL / memory
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /gpfs/home2/dwessels2/code/nvSubquadratic-private
mkdir -p logs

PYTHONPATH=. python experiments/run.py --config "$CONFIG" "$@"
