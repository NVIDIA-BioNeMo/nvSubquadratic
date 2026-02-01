#!/bin/bash
#SBATCH --job-name=nvsubq
#SBATCH --partition=performance
# #SBATCH --partition=capacity
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=128


# Activate venv
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

PYTHONPATH=. python -m experiments.run --config examples/imagenet_classification/tiny_ccnn_7_512_hyena_custom_augs.py train.iterations=1000 dataset.batch_size=8