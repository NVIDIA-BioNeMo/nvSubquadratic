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
# PYTHONPATH=. python experiments/run.py --config examples/text_pretraining/zyda_1d_hyena.py
# PYTHONPATH=. python experiments/run.py --config examples/text_pretraining/zyda_1d_attention.py
# python scripts/evaluate_with_lingua.py \
#     --ckpt_path /home/dwessel/code/nvSubquadratic-private/runs/DW_examples_text_pretraining_zyda_1d_attention_2025-12-04-22-48-58/checkpoints/last.ckpt \
#     --config_path examples/text_pretraining/zyda_1d_attention.py \
#     --tasks arc_easy,hellaswag \
#     --batch_size 8 \
#     --device cuda
python scripts/evaluate_with_lingua.py \
    --ckpt_path /home/dwessel/code/nvSubquadratic-private/runs/DW_examples_text_pretraining_zyda_1d_hyena_2025-12-02-14-05-52/checkpoints/last.ckpt \
    --config_path examples/text_pretraining/zyda_1d_hyena.py \
    --tasks arc_easy,hellaswag \
    --batch_size 8 \
    --device cuda

# ImageNet Experiment
# python experiments/run.py --config examples/imagenet_classification/tiny_ccnn_7_512_hyena_circular.py
# wandb agent equivariance/nvSubquadratic-private-experiments/9bt1u7le
# PYTHONPATH=. python experiments/run.py --config examples/text_pretraining/zyda_1d_attention.py
