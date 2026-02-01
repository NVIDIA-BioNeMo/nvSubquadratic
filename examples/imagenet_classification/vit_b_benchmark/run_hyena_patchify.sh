#!/bin/bash
#SBATCH --job-name=vit_b_hyena_patchify
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
source ~/.bashrc
conda activate nvsubq

# Set up paths
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.

# Run training with Hyena + Patchify (most efficient config)
python experiments/run.py \
    --config examples/imagenet_classification/vit_b_benchmark/hyena.py \
    dataset.batch_size=16 \
    train.iterations=100000
