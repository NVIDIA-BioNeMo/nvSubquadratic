#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --partition=gpu_h100
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch [--job-name=NAME] examples/vit5_imagenet/v5_patch/submit_8gpu.sh <config.py> [extra args...]"
    echo ""
    echo "Examples:"
    echo "  sbatch --job-name=hyena-p16 examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/hyena_patch16.py"
    echo "  sbatch --job-name=attn-p8  examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/attention_patch8.py"
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

export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
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

CONFIG=$(realpath "$CONFIG")

# 8 GPUs — batch_size and accumulate_grad_steps are set in each config
# to maintain effective batch size of 2048.
PYTHONPATH=. torchrun --nproc_per_node=8 experiments/run.py --config "$CONFIG" num_nodes=1 "$@"
