#!/bin/bash
#SBATCH --job-name=hyena-train
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/hyena_train_%j.out
#SBATCH --error=slurm/hyena_train_%j.err

# Hyena Text Pre-training on FineWeb Edu 10BT

set -x

# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

# Set environment variables
export PYTHONPATH="."
export HF_HOME="/ivi/zfs/s0/original_homes/dwessel/data/.hf"
export HF_HUB_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Config file - change this for different runs
CONFIG_FILE="examples/text_pretraining/lingua_hyena_train_test.yaml"

echo "================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-2}"
echo "================================================"

# Run training with torchrun for multi-GPU
torchrun --nproc_per_node=${SLURM_GPUS_ON_NODE:-2} \
    --master_port=29500 \
    lingua_clone/apps/main/train.py \
    --config ${CONFIG_FILE}

echo "Training completed with exit code: $?"
