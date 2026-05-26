#!/bin/bash
# Driver for `bench_vit5_compile.py` — submits to SLURM with the standard ViT-5-Small config.
#SBATCH --job-name=vit5-profile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=low
#SBATCH --gpu-bind=closest
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared"
#SBATCH --container-workdir=/home/dwromero/projects/nvSubquadratic-private
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/vit5_profile_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/vit5_profile_%j.err

set -eo pipefail
source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq
cd /home/dwromero/projects/nvSubquadratic-private
PYTHONPATH=. python benchmarks/vit5_imagenet/bench_vit5_compile.py
