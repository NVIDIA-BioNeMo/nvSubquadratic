#!/bin/bash
#SBATCH --job-name=ablation_1gpu
#SBATCH --account=linuxusers
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%x_%j.out

# 1-GPU ablation template for Phases 2–6
# Effective batch size: 1 GPU × 32 × 4 (grad accum) = 128
# Cluster: hipster (performance partition, RTX 6000 Ada)
#
# Usage:
#   sbatch --job-name=<name> run_1gpu_ablation_hipster.sh <config.py> [overrides...]
#
# Examples:
#   # Phase 2.1: omega_0 = 10
#   sbatch --job-name=phase2_omega10 run_1gpu_ablation_hipster.sh hyena_patchify.py \
#       net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=10
#
#   # Phase 3.3: hidden_dim = 128
#   sbatch --job-name=phase3_hdim128 run_1gpu_ablation_hipster.sh hyena_patchify.py \
#       net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.mlp_hidden_dim=128

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

echo "=== 1-GPU Ablation Run ==="
echo "Config: ${CONFIG_PATH}"
echo "Overrides: ${OVERRIDES}"
echo "Effective BS: 1 GPU × 32 × 4 (accum) = 128"
echo "=========================="

# Run training with gradient accumulation to match effective BS=128
python experiments/run.py \
    --config "${CONFIG_PATH}" \
    train.accumulate_grad_steps=4 \
    ${OVERRIDES}
