#!/bin/bash
#SBATCH --job-name=vit5-val
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
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/vit5_validate_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/vit5_validate_%j.err

set -eo pipefail

set -a
source /home/dwromero/projects/nvSubquadratic-private/.env
set +a
export IMAGENET_PATH=/shared/data/image_datasets/imagenet
export SLURM_JOB_NAME=bash

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

cd /home/dwromero/projects/nvSubquadratic-private
PYTHONPATH=. python -u benchmarks/vit5_imagenet/validate_checkpoint.py
