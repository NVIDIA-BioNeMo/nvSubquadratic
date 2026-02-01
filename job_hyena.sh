#!/bin/bash
#SBATCH --job-name=hyena
#SBATCH --partition=capacity
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --cpus-per-task=128

# Activate venv
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export HF_TOKEN=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
export KAGGLE_KEY=a370774e0be6fff63358e4febdeac5a8
export KAGGLE_USERNAME=dafidofff
export PYTHONPATH=.

# Disable tokenizers parallelism to avoid segfaults
export TOKENIZERS_PARALLELISM=false

# Run Hyena training with Lingua (matching transformer run settings)
PYTHONPATH=lingua_clone:. torchrun --nproc_per_node=4 lingua_clone/apps/main/train.py \
    config=examples/text_pretraining/lingua_hyena_train_balanced.yaml \
    dump_dir=results/hyena_fineweb10bt_lingua
