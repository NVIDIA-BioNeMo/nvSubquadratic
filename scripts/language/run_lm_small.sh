#!/bin/bash
#SBATCH --job-name=lm-small
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm/lm_small_%j.out
#SBATCH --error=slurm/lm_small_%j.err

# LM Small tier — 4x RTX 3090, ~2-4 hours
# Usage:
#   sbatch scripts/language/run_lm_small.sh examples/language_modeling/small_hyena.py
#   sbatch scripts/language/run_lm_small.sh examples/language_modeling/small_attention.py

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
echo "GPUs: 4 (RTX 3090)"
echo "Tier: SMALL (~25M params)"
echo "================================================"

python experiments/run.py --config ${CONFIG_FILE}

echo "Training completed with exit code: $?"
