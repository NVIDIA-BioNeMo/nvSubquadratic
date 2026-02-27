#!/bin/bash
#SBATCH --job-name=profile-bottleneck
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --partition=low
#SBATCH --exclude=b65c909e-02,b65c909e-05
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared,/scratch:/scratch"
#SBATCH --container-workdir=/home/dwromero/projects/nvSubquadratic-private
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/profile_bottleneck_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/profile_bottleneck_%j.err

set -eo pipefail
set -a
source /home/dwromero/projects/nvSubquadratic-private/.env
set +a
export IMAGENET_PATH="${IMAGENET_PATH:-/shared/data/image_datasets/imagenet}"
export IMAGENET_FOLDER_PATH="${IMAGENET_FOLDER_PATH:-/shared/data/image_datasets/imagenet_folder}"
export WANDB_MODE=disabled
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR=/home/dwromero/.triton/cache

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

cd /home/dwromero/projects/nvSubquadratic-private

# Triton autotuning needs ldconfig; create symlink if missing in container
if [ ! -f /sbin/ldconfig ] && command -v ldconfig &>/dev/null; then
    mkdir -p /sbin && ln -sf "$(command -v ldconfig)" /sbin/ldconfig
fi

NGPUS=$(nvidia-smi -L | wc -l)

if echo "$@" | grep -q -- "--ddp"; then
    PYTHONPATH=. torchrun --nproc_per_node="$NGPUS" benchmarks/vit5_imagenet/profile_training_bottleneck.py "$@"
else
    PYTHONPATH=. python benchmarks/vit5_imagenet/profile_training_bottleneck.py "$@"
fi
