#!/bin/bash
#SBATCH --account=all6000users
#SBATCH --partition=all6000
#SBATCH --gpus=1
#SBATCH --job-name=arc_vit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=3-00:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/arc_%A.out

# print assigned node
echo "This job is running on node: $SLURM_NODELIST"

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch [--job-name=NAME] examples/arc/submit_arc_1gpu_a6000.sh <config.py> [extra args...]"
    echo "  e.g. sbatch examples/arc/submit_arc_1gpu_a6000.sh examples/arc/cfg_vit.py"
    exit 1
fi

CONFIG="$1"
shift

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export WANDB_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb
export WANDB_DATA_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb

# CUDA (needed by torch.compile / inductor)
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

# Memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /home/dwessel/code/nvSubquadratic-private
mkdir -p logs

PYTHONPATH=. python experiments/run.py --config "$CONFIG" "$@"
