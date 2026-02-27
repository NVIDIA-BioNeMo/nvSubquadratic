#!/bin/bash
#SBATCH --job-name=bench_dali
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/%x_%j.out

# Head-to-head benchmark: ImageFolder vs DALI on single GPU.
#
# image_size=224, final_image_size=224 matches the training config in
# examples/vit5_imagenet/vit5_small_pretrain_apex.py.
#
# Uses 14 workers/threads to match the training worker-to-GPU ratio
# (8 GPUs × 14 workers = 112 CPUs total).
#
# Runs twice: once without augmentations (pure decode speed) and once
# with the full training augmentations (ColorJitter + ThreeAugment).

set -eo pipefail

# Setup env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# Determine ImageFolder path (use local scratch if extracted, else copy from source node)
export IMAGENET_FOLDER_PATH="${IMAGENET_FOLDER_PATH:-/local_scratch/$USER/imagenet_folder}"
SOURCE_NODE="ivi-cn020"
SOURCE_PATH="/local_scratch/dknigge/imagenet_folder"

if [ ! -d "$IMAGENET_FOLDER_PATH/train" ]; then
    echo "ImageNet not found locally, copying from ${SOURCE_NODE}:${SOURCE_PATH} ..."
    echo "This may take a while on first run."
    mkdir -p "$IMAGENET_FOLDER_PATH"
    rsync -a --info=progress2 "${SOURCE_NODE}:${SOURCE_PATH}/" "$IMAGENET_FOLDER_PATH/"
    echo "Copy complete."
fi
if [ ! -d "$IMAGENET_FOLDER_PATH/train" ]; then
    echo "ERROR: ImageNet not found at $IMAGENET_FOLDER_PATH"
    exit 1
fi

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "ImageNet path: $IMAGENET_FOLDER_PATH"
echo "=========================================================="

BASE_ARGS=(
    --imagefolder-dir "$IMAGENET_FOLDER_PATH"
    --batch-size 256
    --num-workers 14
    --num-batches 200
    --image-size 224
    --final-image-size 224
)

echo ""
echo "##########################################################"
echo "# Run 1: basic augmentations (decode + crop + flip only) #"
echo "##########################################################"
python scripts/benchmark_dali_vs_folder.py "${BASE_ARGS[@]}"

echo ""
echo "##########################################################"
echo "# Run 2: full training augmentations (+ ColorJitter + ThreeAugment)"
echo "##########################################################"
python scripts/benchmark_dali_vs_folder.py "${BASE_ARGS[@]}" \
    --three-augment \
    --color-jitter 0.3

echo "Benchmark complete at $(date)"
