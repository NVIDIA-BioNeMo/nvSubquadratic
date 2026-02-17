#!/bin/bash
#SBATCH --job-name=phase0_vit_b_attn_patch4
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%x_%j.out

# Phase 0.1: ViT-B + patch-4 baseline (pipeline validation)
# Effective batch size: 4 GPUs × 32 = 128

# Activate environment
source ~/.bashrc
conda activate nvsubq

# Set up paths
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.
source .env

# Set dataset cache paths
export TINYIMAGENET_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/tinyimagenet'
export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf'
export HF_HUB_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub'

# Run training
python experiments/run.py \
    --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/attention_patchify.py
