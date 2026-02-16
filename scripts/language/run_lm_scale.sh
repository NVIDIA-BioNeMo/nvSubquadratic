#!/bin/bash
#SBATCH --job-name=lm-scale
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/lm_scale_%j.out
#SBATCH --error=slurm/lm_scale_%j.err

# LM Scale tier — 32x A100 (4 nodes x 8 GPUs), ~1-2 days
# Placeholder for future scaling experiments.
# Usage:
#   sbatch scripts/language/run_lm_scale.sh examples/language_modeling/scale_hyena.py
#   sbatch scripts/language/run_lm_scale.sh examples/language_modeling/scale_attention.py

set -x

CONFIG_FILE="${1:-examples/language_modeling/scale_hyena.py}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export HF_HOME="/ivi/zfs/s0/original_homes/dwessel/data/.hf"
export HF_HUB_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

echo "================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_FILE}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 8"
echo "Tier: SCALE (~350M params)"
echo "================================================"

# Multi-node: use srun with torchrun
srun torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${SLURM_NODELIST}:29500 \
    experiments/run.py --config ${CONFIG_FILE}

echo "Training completed with exit code: $?"
