#!/bin/bash
#SBATCH --job-name=vit_b_hyena_patchify
#SBATCH --account=all6000users
#SBATCH --partition=all6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out

# Activate environment
source ~/.bashrc
conda activate nvsubq

# Set up paths
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.

# Run training with Hyena + Patchify (most efficient config)
python experiments/run.py \
    --config examples/imagenet_classification/vit_b_benchmark/hyena_patchify.py \
    dataset.batch_size=16 \
    train.iterations=100000
