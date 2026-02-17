#!/bin/bash
#SBATCH --job-name=attn-16x16
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/attn_16x16_%j.out

# Imagenette ViT-B benchmark - Attention 16x16 patches (160px)
# Usage:
#   sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_attn_16x16.sh

set -x

CONFIG_FILE="examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/imagenette_attention_patch16.py"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export IMAGENETTE_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/imagenette"
export HF_HOME="/ivi/zfs/s0/original_homes/dwessel/data/.hf"
export HF_HUB_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub"

python experiments/run.py --config ${CONFIG_FILE} "$@"
