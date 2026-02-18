#!/bin/bash
#SBATCH --job-name=phase0_hyena_patch4
#SBATCH --account=linuxusers
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%x_%j.out

# Phase 0.2: Hyena + patch-4 baseline (pipeline validation)
# Effective batch size: 4 GPUs × 32 = 128
# Cluster: hipster (performance partition, RTX 6000 Ada)

# Activate environment
source ~/.bashrc
conda activate nvsubq

# Set up paths
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.

# Run training
python experiments/run.py \
    --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/hyena_patchify.py
