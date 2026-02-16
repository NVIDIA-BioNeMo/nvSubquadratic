#!/bin/bash
#SBATCH --job-name=lm-medium
#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/lm_medium_%j.out
#SBATCH --error=slurm/lm_medium_%j.err

# LM Medium tier — 2x A6000 (limited), ~16-24 hours
# Usage:
#   sbatch scripts/language/run_lm_medium.sh examples/language_modeling/medium_hyena.py
#   sbatch scripts/language/run_lm_medium.sh examples/language_modeling/medium_attention.py

set -x

CONFIG_FILE="${1:-examples/language_modeling/medium_hyena.py}"

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
echo "GPUs: 2 (A6000)"
echo "Tier: MEDIUM (~125M params)"
echo "================================================"

shift 1
python experiments/run.py --config ${CONFIG_FILE} "$@"

echo "Training completed with exit code: $?"
