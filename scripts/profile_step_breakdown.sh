#!/bin/bash
#SBATCH --job-name=step-breakdown
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
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/step_breakdown_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/step_breakdown_%j.err

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

# Triton autotuning needs ldconfig; create symlink or stub if missing
if [ ! -f /sbin/ldconfig ]; then
    mkdir -p /sbin 2>/dev/null || true
    LDCONFIG_PATH=$(which ldconfig 2>/dev/null || true)
    if [ -n "$LDCONFIG_PATH" ]; then
        ln -sf "$LDCONFIG_PATH" /sbin/ldconfig 2>/dev/null || true
    else
        printf '#!/bin/sh\n' > /sbin/ldconfig 2>/dev/null && chmod +x /sbin/ldconfig 2>/dev/null || true
    fi
fi

NGPUS=$(nvidia-smi -L | wc -l)

if echo "$@" | grep -q -- "--ddp"; then
    PYTHONPATH=. torchrun --nproc_per_node="$NGPUS" scripts/profile_step_breakdown.py "$@"
else
    PYTHONPATH=. python scripts/profile_step_breakdown.py "$@"
fi
