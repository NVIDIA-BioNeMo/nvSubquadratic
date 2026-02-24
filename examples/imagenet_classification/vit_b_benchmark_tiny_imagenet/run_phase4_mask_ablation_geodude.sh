#!/bin/bash
#SBATCH --job-name=phase4_mask_ablation
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=slurm/%x_%j.out

# Phase 4: Mask Ablation (Hyena + patch-4) — Sequential runs on geodude
# Effective batch size: 4 GPUs × 32 = 128
# Phase 4.1 (Gaussian mask) = Phase 0.2 baseline (Job 174875, 70.67%) — no need to re-run
# Runs: 4.2 No mask → 4.3 Exponential mask

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

CONFIG_DIR="examples/imagenet_classification/vit_b_benchmark_tiny_imagenet"

echo "=========================================="
echo "Phase 4.2: No mask (Identity)"
echo "=========================================="
python experiments/run.py \
    --config ${CONFIG_DIR}/hyena_patchify_no_mask.py

echo "=========================================="
echo "Phase 4.3: Exponential mask"
echo "=========================================="
python experiments/run.py \
    --config ${CONFIG_DIR}/hyena_patchify_exp_mask.py
