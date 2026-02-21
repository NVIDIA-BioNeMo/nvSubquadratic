#!/bin/bash
#SBATCH --job-name=profile-bottleneck
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=low
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared"
#SBATCH --container-workdir=/home/dwromero/projects/nvSubquadratic-private
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/profile_bottleneck_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/profile_bottleneck_%j.err

set -eo pipefail
set -a
source /home/dwromero/projects/nvSubquadratic-private/.env
set +a
export IMAGENET_PATH=/shared/data/image_datasets/imagenet
export IMAGENET_FOLDER_PATH=/shared/data/image_datasets/imagenet_folder

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR=/home/dwromero/.triton/cache

cd /home/dwromero/projects/nvSubquadratic-private
PYTHONPATH=. python scripts/profile_training_bottleneck.py
