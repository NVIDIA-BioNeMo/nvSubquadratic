#!/bin/bash
#SBATCH --job-name=imagenette-hyena
#SBATCH --partition=geodude
#SBATCH --account=geodudeusers
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/imagenette_hyena%j.out

# Imagenette ViT-B benchmark (160px)
# Usage:
#   sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_imagenette_hyena_patchify.sh

set -x

CONFIG_FILE="examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/imagenette_hyena_patchify.py"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export IMAGENETTE_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/imagenette"
export HF_HOME="/ivi/zfs/s0/original_homes/dwessel/data/.hf"
export HF_HUB_CACHE="/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub"

python experiments/run.py --config ${CONFIG_FILE} "$@"
