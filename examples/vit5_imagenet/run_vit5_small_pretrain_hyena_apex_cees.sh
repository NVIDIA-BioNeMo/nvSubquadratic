#!/bin/bash
#SBATCH --job-name=vit5_hyena_apex
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=96:00:00
#SBATCH --output=slurm/%x_%j.out

# ViT-5-Small + Hyena (Apex FusedLAMB) on ImageNet-1K
# Cluster: cees (8× RTX A5000 24 GB, 7-day limit)
#
# Effective batch size: 8 GPUs × 128 batch/GPU × 2 accum steps = 2048
# Architecture: ViT-5-Small, hidden_dim=384, 12 blocks, patch_size=16
# Sequence mixer: 2D Hyena (CKConvND + SIREN kernel)

set -euo pipefail

# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

# Set up paths
cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)  # WandB API key + HF_TOKEN

# Dataset paths — all rooted under the repo's data/ symlink
# data/ -> /ivi/zfs/s0/original_homes/dknigge  (shared ZFS, accessible from cees)
export IMAGENET_PATH="data/imagenet"
export IMAGENET_FOLDER_PATH="data/imagenet_folder"
export HF_HOME="data/.hf"
export HF_HUB_CACHE="data/.hf/hub"

# Run training — 8 GPUs × 128 × 2 accum = 2048 effective batch size
python experiments/run.py \
    --config examples/vit5_imagenet/vit5_small_pretrain_hyena_apex.py
