#!/bin/bash
#SBATCH --job-name=extract-imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --partition=low
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared"
#SBATCH --container-workdir=/home/dwromero/projects/nvSubquadratic-private
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/extract_imagenet_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/extract_imagenet_%j.err

set -eo pipefail
set -a
source /home/dwromero/projects/nvSubquadratic-private/.env
set +a
export IMAGENET_PATH=/shared/data/image_datasets/imagenet

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

cd /home/dwromero/projects/nvSubquadratic-private
PYTHONPATH=. python scripts/extract_imagenet_to_folder.py
