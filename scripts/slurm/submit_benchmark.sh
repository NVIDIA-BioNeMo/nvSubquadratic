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
#SBATCH --time=02:00:00

set -eo pipefail

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${USER}_torch2.10
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}_${SLURM_JOB_ID}

cd /home/dwromero/projects/nvSubquadratic-private

if [ -z "$TRITON_LIBCUDA_PATH" ]; then
    _libcuda=$(find /usr/lib /usr/local/lib /usr/lib64 /lib /lib64 -name "libcuda.so.1" 2>/dev/null | head -1 || true)
    if [ -n "$_libcuda" ]; then
        export TRITON_LIBCUDA_PATH="$(dirname "$_libcuda")"
    fi
fi

PYTHONPATH=. python "$@"
