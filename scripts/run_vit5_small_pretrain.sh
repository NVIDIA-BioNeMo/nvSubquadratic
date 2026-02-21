#!/bin/bash
#SBATCH --job-name=vit5-small-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --partition=low
#SBATCH --gpu-bind=closest
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared"
#SBATCH --container-workdir=/home/dwromero/projects/nvSubquadratic-private
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/vit5_small_pretrain_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/vit5_small_pretrain_%j.err

set -eo pipefail

# Source environment variables (WandB, HF tokens)
set -a
source /home/dwromero/projects/nvSubquadratic-private/.env
set +a
export IMAGENET_PATH=/shared/data/image_datasets/imagenet

# Activate conda from the home directory (mounted into the container)
source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

# Prevent Lightning from auto-detecting SLURM (ntasks-per-node=1 with 8 GPUs
# conflicts with Lightning's SLURMEnvironment validation). Setting JOB_NAME to
# "bash" is Lightning's documented way to disable SLURM environment detection,
# letting it spawn its own DDP subprocesses via the subprocess launcher.
export SLURM_JOB_NAME=bash

# Run training
cd /home/dwromero/projects/nvSubquadratic-private
PYTHONPATH=. python experiments/run.py \
    --config examples/vit5_imagenet/vit5_small_pretrain.py
