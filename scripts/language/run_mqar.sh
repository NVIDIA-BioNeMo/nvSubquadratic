#!/bin/bash
#SBATCH --job-name=mqar
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/mqar_%j.out
#SBATCH --error=slurm/mqar_%j.err

# MQAR Associative Recall — Hyena vs Attention
# Usage:
#   sbatch scripts/language/run_mqar.sh examples/mqar/hyena_causal.py
#   sbatch scripts/language/run_mqar.sh examples/mqar/attention_causal.py

set -x

# Config file from argument or default
CONFIG_FILE="${1:-examples/mqar/hyena_causal.py}"

# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

# Environment
export PYTHONPATH="."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: 1"
echo "================================================"

shift 1
python experiments/run.py --config ${CONFIG_FILE} "$@"

echo "MQAR training completed with exit code: $?"
