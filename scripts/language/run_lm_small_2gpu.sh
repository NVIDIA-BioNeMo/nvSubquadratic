#!/bin/bash
#SBATCH --job-name=lm-small-2gpu
#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/lm_small_2gpu_%j.out
#SBATCH --error=slurm/lm_small_2gpu_%j.err

# LM Small tier — 2x GPU (all6000), ~2-4 hours
# Note: Uses gradient accumulation = 2 to match effective batch size of 4-GPU run.

set -x

CONFIG_FILE="${1:-examples/language_modeling/small_hyena.py}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export HF_HOME="/ivi/zfs/s0/original_homes/dwessel/data/.hf"
export HF_HUB_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 2 (Limit on all6000 is 2)"
echo "Tier: SMALL (~25M params)"
echo "Mode: Gradient Accumulation = 2"
echo "================================================"

# Override accumulate_grad_steps to 2 to compensate for halving the GPU count (4 -> 2)
python experiments/run.py --config ${CONFIG_FILE} train.accumulate_grad_steps=2
