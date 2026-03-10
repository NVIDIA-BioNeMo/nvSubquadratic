#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir="/gpfs/home1/dknigge/nvsq_update_branch"
#SBATCH --output=./runs/slurm/pack_imagenet_%j.out
#SBATCH --error=./runs/slurm/pack_imagenet_%j.err
#SBATCH --job-name='pack-imagenet-tar'

source /home/dknigge/.bashrc
source /gpfs/home1/dknigge/nvsq_update_branch/.venv/bin/activate

set -euo pipefail

echo "[pack] Node: $(hostname)"
echo "[pack] Starting at $(date)"

export IMAGENET_PATH="/home/dknigge/project_dir/huggingface/imagenet"
export IMAGENET_OUTPUT_TAR="/scratch-shared/dknigge/imagenet_imagefolder.tar"
export PYTHONPATH="/gpfs/home1/dknigge/nvsq_update_branch:${PYTHONPATH:-}"

python -u scripts/extract_imagenet_to_tar.py

echo "[pack] Finished at $(date)"
