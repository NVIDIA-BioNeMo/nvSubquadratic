#!/bin/bash
#SBATCH --job-name=phase1_hyena_patch4_rff
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%x_%j.out

# Phase 1.2: Hyena + patch-4 with RFF Kernel baseline on geodude (4 GPUs)
# Effective Batch Size: 4 GPUs * 32 bs/gpu * 1 accum = 128

source ~/.bashrc
conda activate nvsubq
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.
source .env

# Set dataset cache paths
export TINYIMAGENET_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/tinyimagenet'
export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf'
export HF_HUB_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub'
python experiments/run.py --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/hyena_patchify_rff.py
