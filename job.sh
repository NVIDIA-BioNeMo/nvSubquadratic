#!/bin/bash
#SBATCH --job-name=nvsubq
#SBATCH --partition=performance
# #SBATCH --partition=capacity
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --cpus-per-task=64

# Activate venv
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export HF_TOKEN=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
export KAGGLE_KEY=a370774e0be6fff63358e4febdeac5a8
export KAGGLE_USERNAME=dafidofff
export PYTHONPATH=.

# Language Experiment
export TOKENIZERS_PARALLELISM=false    # Disable tokenizers parallelism to avoid segfaults

PYTHONPATH=lingua_clone:. torchrun --nproc_per_node=2 lingua_clone/apps/main/train.py \
    config=examples/text_pretraining/lingua_hyena_train_balanced.yaml \
    dump_dir=results/hyena_fineweb10bt_lingua_balanced