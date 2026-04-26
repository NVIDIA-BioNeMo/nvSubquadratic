#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH --gpu-bind=closest
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared,/scratch:/scratch,/dev/shm:/dev/shm"
#SBATCH --container-workdir=/home/dwromero/projects/nvSubquadratic-private
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/%x_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/%x_%j.err

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch [--job-name=NAME] scripts/submit_1gpu.sh <config.py> [extra args...]"
    echo "  e.g. sbatch --job-name=vit5-apex scripts/submit_1gpu.sh examples/vit5_imagenet/vit5_small_pretrain_apex.py"
    exit 1
fi

CONFIG="$1"
shift

set -a
source /home/dwromero/projects/nvSubquadratic-private/.env
set +a
export IMAGENET_PATH=/shared/data/image_datasets/imagenet
export IMAGENET_FOLDER_PATH=/shared/data/image_datasets/imagenet_folder

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

export SLURM_JOB_NAME=bash

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${USER}_torch2.10
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}_${SLURM_JOB_ID}
export DALI_NO_MMAP=1

cd /home/dwromero/projects/nvSubquadratic-private

# Triton calls /sbin/ldconfig to find libcuda — bypass entirely via env knob
if [ -z "$TRITON_LIBCUDA_PATH" ]; then
    _libcuda=$(find /usr/lib /usr/local/lib /usr/lib64 /lib /lib64 -name "libcuda.so.1" 2>/dev/null | head -1 || true)
    if [ -n "$_libcuda" ]; then
        export TRITON_LIBCUDA_PATH="$(dirname "$_libcuda")"
    fi
fi

PYTHONPATH=. python experiments/run.py --config "$CONFIG" "$@"
