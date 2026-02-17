#!/bin/bash
#SBATCH --job-name=vit_b_hyena
#SBATCH --partition=capacity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%x_%j.out

# Activate environment
source ~/.bashrc
conda activate nvsubq

# Set up paths
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.

# Run training with Hyena (no patchification)
python experiments/run.py \
    --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/hyena.py
