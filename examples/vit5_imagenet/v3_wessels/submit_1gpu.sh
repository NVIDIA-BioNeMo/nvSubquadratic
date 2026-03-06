#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch [--job-name=NAME] examples/vit5_imagenet/v3/submit_1gpu.sh <config.py> [extra args...]"
    echo "  e.g. sbatch --job-name=v3-gated examples/vit5_imagenet/v3/submit_1gpu.sh examples/vit5_imagenet/v3/vit5_small_pretrain_hyena_cls_row_apex_gated.py"
    exit 1
fi

CONFIG="$1"
shift

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export IMAGENET_PATH=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageNet/
export IMAGENET_FOLDER_PATH=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder
export LOCAL_STAGING_DIR=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR=~/.triton/cache_${SLURM_JOB_ID}
export DALI_NO_MMAP=1

# CUDA module
module load 2025
module load CUDA/12.8.0

# NCCL / memory
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /gpfs/home2/dwessels2/code/nvSubquadratic-private
mkdir -p logs

# 1 GPU × 256 batch_size × 8 accum = 2048 effective batch size (matches 8-GPU run)
# Disable CUDA graphs (compile_mode=None) — max-autotune CUDA graphs are incompatible
# with gradient accumulation (tensor buffer overwrites between micro-steps).
# torch.compile still applies kernel fusion/codegen, just without graph capture.
PYTHONPATH=. python experiments/run.py --config "$CONFIG" train.accumulate_grad_steps=8 compile_mode=max-autotune-no-cudagraphs "$@"