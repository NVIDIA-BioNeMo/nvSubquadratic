#!/bin/bash
#SBATCH --job-name=hyena_vit5
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=96:00:00
#SBATCH --output=slurm/%x_%j.out

set -euo pipefail

# Setup env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# Force use of WebDataset path
export IMAGENET_WDS_PATH="data/imagenet-wds"

# DDP Configuration
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID

# Run training
python experiments/run.py \
    --config-path examples/vit5_imagenet/vit5_small_hyena_pretrain.py
