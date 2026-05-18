#!/bin/bash
#SBATCH --account=all6000users
#SBATCH --partition=all6000
#SBATCH --gpus=1
#SBATCH --job-name=arc_profile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=0-02:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/arc_profile_%A.out

# print assigned node
echo "This job is running on node: $SLURM_NODELIST"

set -eo pipefail

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

# Default: ViT config, compiled mode
# Pass --eager for no torch.compile
# Pass --no-rearc to skip RE-ARC data (faster startup, ~13k instead of ~413k train examples)
# Pass --config examples/arc/cfg_hyena_rearc.py to profile the Hyena backbone
CONFIG="${1:-examples/arc/cfg_vit_rearc.py}"
shift || true  # remaining args forwarded to the profiling script

PYTHONPATH=. python benchmarks/arc/profile_training_bottleneck.py \
    --config "$CONFIG" \
    "$@"
