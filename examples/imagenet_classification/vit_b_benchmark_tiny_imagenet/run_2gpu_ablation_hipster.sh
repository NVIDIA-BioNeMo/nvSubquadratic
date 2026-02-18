#!/bin/bash
#SBATCH --job-name=ablation_2gpu
#SBATCH --account=linuxusers
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%x_%j.out

# 2-GPU ablation template for Phases 2–6
# Effective batch size: 2 GPUs × 32 × 2 (grad accum) = 128
# Cluster: hipster (capacity partition, L4)
#
# Usage:
#   sbatch --job-name=<name> run_2gpu_ablation_hipster.sh <config.py> [overrides...]
#
# Examples:
#   # Phase 2.1: omega_0 = 10
#   sbatch --job-name=phase2_omega10 run_2gpu_ablation_hipster.sh hyena_patchify.py \
#       net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=10

CONFIG=${1:?Error: provide config file basename as first argument}
shift
OVERRIDES="$@"

# Activate environment
source ~/.bashrc
conda activate nvsubq

# Set up paths
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.

CONFIG_PATH="examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/${CONFIG}"

echo "=== 2-GPU Ablation Run ==="
echo "Config: ${CONFIG_PATH}"
echo "Overrides: ${OVERRIDES}"
echo "Effective BS: 2 GPUs × 32 × 2 (accum) = 128"
echo "=========================="

# Run training with gradient accumulation to match effective BS=128
# With 2 GPUs, we need 2 accum steps to get 128 (2 GPUs * 32 per GPU * 2 accum = 128)
python experiments/run.py \
    --config "${CONFIG_PATH}" \
    train.accumulate_grad_steps=2 \
    ${OVERRIDES}
