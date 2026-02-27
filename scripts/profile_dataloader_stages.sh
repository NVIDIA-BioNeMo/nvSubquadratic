#!/bin/bash
#SBATCH --job-name=profile_dl_stages
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/%x_%j.out

# Profile the dataloading pipeline stage-by-stage.
# Uses 1 GPU and 16 CPUs to match the per-GPU ratio of 8-GPU training
# (8 GPUs × 13 CPUs/GPU ≈ 104 CPUs total, so ~13-16 CPUs per GPU).

set -eo pipefail

# Setup env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# Use local scratch if available; fall back to shared path.
export IMAGENET_FOLDER_PATH="${IMAGENET_FOLDER_PATH:-/local_scratch/$USER/imagenet_folder}"
if [ ! -d "$IMAGENET_FOLDER_PATH/train" ]; then
    echo "ImageNet not found at $IMAGENET_FOLDER_PATH, trying shared path..."
    export IMAGENET_FOLDER_PATH="/shared/data/image_datasets/imagenet_folder"
fi
if [ ! -d "$IMAGENET_FOLDER_PATH/train" ]; then
    echo "ERROR: ImageNet not found. Set IMAGENET_FOLDER_PATH to the ImageFolder directory."
    exit 1
fi
echo "Using ImageNet at: $IMAGENET_FOLDER_PATH"

echo "Starting dataloading stage profiler..."
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python scripts/profile_dataloader_stages.py \
    --data-dir "$IMAGENET_FOLDER_PATH" \
    --num-images 500 \
    --num-batches 100 \
    --batch-size 256 \
    --json slurm/profile_dl_stages_${SLURM_JOB_ID}.json

echo "Profiling complete!"
