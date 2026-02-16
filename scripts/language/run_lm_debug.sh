#!/bin/bash
#SBATCH --job-name=lm-debug
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/lm_debug_%j.out
#SBATCH --error=slurm/lm_debug_%j.err

# LM Debug tier — 1 GPU, ~5 minutes
# Usage:
#   sbatch scripts/language/run_lm_debug.sh examples/language_modeling/debug_hyena.py
#   sbatch scripts/language/run_lm_debug.sh examples/language_modeling/debug_attention.py

set -x

CONFIG_FILE="${1:-examples/language_modeling/debug_hyena.py}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export HF_HOME="/ivi/zfs/s0/original_homes/dwessel/data/.hf"
export HF_HUB_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 1"
echo "Tier: DEBUG"
echo "================================================"

shift 1
python experiments/run.py --config ${CONFIG_FILE} "$@"

echo "Training completed with exit code: $?"
