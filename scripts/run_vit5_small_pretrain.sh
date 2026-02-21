#!/bin/bash
#SBATCH --job-name=vit5-small-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
# Must match or exceed (num_gpus × num_workers_per_gpu) to avoid CPU
# oversubscription.  Lightning DDP spawns 8 processes, each with 16 workers
# = 128 total.  If you set this to 16, workers fight for CPUs and data
# loading becomes the bottleneck (124ms → ~1000ms per batch).
#SBATCH --cpus-per-task=128
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
export IMAGENET_FOLDER_PATH=/shared/data/image_datasets/imagenet_folder

# Activate conda from the home directory (mounted into the container)
source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

# Prevent Lightning from auto-detecting SLURM (ntasks-per-node=1 with 8 GPUs
# conflicts with Lightning's SLURMEnvironment validation). Setting JOB_NAME to
# "bash" is Lightning's documented way to disable SLURM environment detection,
# letting it spawn its own DDP subprocesses via the subprocess launcher.
export SLURM_JOB_NAME=bash

# Cache torch.compile / Triton autotuned kernels to disk so restarts skip
# the ~5 min warmup.  The cache is keyed on graph structure + GPU type,
# so it's safe across resume / preemption on the same hardware.
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR=/home/dwromero/.triton/cache

# Run training
cd /home/dwromero/projects/nvSubquadratic-private
PYTHONPATH=. python experiments/run.py \
    --config examples/vit5_imagenet/vit5_small_pretrain.py
