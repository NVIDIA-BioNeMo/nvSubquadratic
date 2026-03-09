#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch [--job-name=NAME] examples/vit5_imagenet/v3/submit_8gpu.sh <config.py> [extra args...]"
    echo "  e.g. sbatch --job-name=v3-gated examples/vit5_imagenet/v3/submit_8gpu.sh examples/vit5_imagenet/v3/vit5_small_pretrain_hyena_cls_row_apex_gated.py"
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

# CUDA module — run in the batch script so LD_LIBRARY_PATH is captured by --export=ALL
module load 2025
module load CUDA/12.8.0

# NCCL / memory
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /gpfs/home2/dwessels2/code/nvSubquadratic-private
mkdir -p logs

CONFIG=$(realpath "$CONFIG")

# 8 GPUs × 256 batch_size = 2048 effective batch size — no gradient accumulation needed.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

# --gpus-per-node=4 is required on the srun step so SLURM sets CUDA_VISIBLE_DEVICES
# correctly on each node (without it the step gets no GPU binding even though the
# job allocation has GPUs). --export=ALL propagates the conda/CUDA env from above.
srun --gpus-per-node=4 --export=ALL bash -c "
    export PYTHONPATH=.
    torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --rdzv_id=\"$SLURM_JOB_ID\" \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\"$MASTER_ADDR:$MASTER_PORT\" \
        experiments/run.py --config \"$CONFIG\" num_nodes=2 \"\$@\"
" bash "$@"