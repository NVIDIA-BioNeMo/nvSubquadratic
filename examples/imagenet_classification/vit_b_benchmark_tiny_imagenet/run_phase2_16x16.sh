#!/bin/bash
#SBATCH --job-name=phase2-16x16
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/phase2_16x16_%j.out

# Phase 2: 16x16 Patch Experiments (100 tokens)
# Usage:
#   sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_phase2_16x16.sh

set -x

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export IMAGENETTE_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/imagenette"
export HF_HOME="/ivi/zfs/s0/original_homes/dwessel/data/.hf"
export HF_HUB_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub"

# Launch BOTH Attention and Hyena sequentially in the same job or parallel if resources allow?
# Slurm job has 2 GPUs. I can launch each on 1 GPU manually if I want them parallel.
# Or just run them one after another. Since time is 12h, sequential is safer.

echo "Running Attention (16x16 patches)..."
python experiments/run.py --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/imagenette_attention_patch16.py

echo "Running Hyena (16x16 patches)..."
python experiments/run.py --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/imagenette_hyena_patch16.py
