#!/bin/bash
#SBATCH --job-name=nvsubq
# #SBATCH --partition=performance
#SBATCH --partition=capacity
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=32

# Activate venv
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export HF_TOKEN=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
export KAGGLE_KEY=a370774e0be6fff63358e4febdeac5a8
export KAGGLE_USERNAME=dafidofff
export PYTHONPATH=.

# Language Experiment
export TOKENIZERS_PARALLELISM=false # Disable tokenizers parallelism to avoid segfaults
# python scripts/prepare_lingua_data.py \
#     --dataset_name Zyphra/Zyda-2 \
#     --output_dir data/lingua_zyda \
#     --chunk_size 10000
python lingua_clone/setup/download_prepare_hf_data.py fineweb_edu_10bt 24 \
    --data_dir ./data \
    --seed 42 \
    --nchunks 16