#!/bin/bash
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gpus=2
#SBATCH --job-name=run_node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=2-00:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/the_well_%A.out

# print assigned node
echo "This job is running on node: $SLURM_NODELIST"

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

export WANDB_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb
export WANDB_DATA_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb
export WELL_DATA_PATH="${WELL_DATA_PATH:-/ivi/zfs/s0/original_homes/dwessel/data/the_well/datasets}"

# CUDA (needed by torch.compile / inductor)
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

# NCCL / memory
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /home/dwessel/code/nvSubquadratic-private
mkdir -p logs

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 experiments/run.py --config "$CONFIG" "$@"
