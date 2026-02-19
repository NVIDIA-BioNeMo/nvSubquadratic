#!/bin/bash
#SBATCH --job-name=phase0_vit_b16_imagenet1k
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=96:00:00
#SBATCH --output=slurm/%x_%j.out

# Phase 0.3: ViT-B/16 attention + patchify on full ImageNet-1K (pipeline sanity check)
# Cluster: cees (8× RTX A5000 24 GB, 7-day limit)
#
# Effective batch size: 8 GPUs × 128 = 1024  (DeiT-B standard)
# Sequence length: 224/16 = 14×14 = 196 tokens
# Expected throughput: ~3–4 it/s  →  300k iters ≈ 20–28 h
# Expected val acc (300k iters ≈ 240 epochs): ≥ 70% top-1

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
# First run will download ILSVRC/imagenet-1k via HuggingFace (~140 GB) into data/imagenet
export IMAGENET_CACHE="data/imagenet"
export HF_HOME="data/.hf"
export HF_HUB_CACHE="data/.hf/hub"

# Run training — no gradient accumulation needed: 8 GPUs × 128 = 1024 eff. BS
python experiments/run.py \
    --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/attention_patchify_imagenet1k.py
