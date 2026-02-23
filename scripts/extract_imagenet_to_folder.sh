#!/bin/bash
#SBATCH --job-name=extract-imagenet
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/extract_imagenet_%j.out

set -eo pipefail

# Setup env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# Extraction
python scripts/extract_imagenet_to_folder.py
