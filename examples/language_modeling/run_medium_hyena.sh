#!/bin/bash
#SBATCH --job-name=lm_medium_hyena
#SBATCH --account=all6000users
#SBATCH --partition=all6000
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/%x_%j.out

# Tier 3 — Medium: Hyena causal LM on WikiText-103 (~125M params)
# Effective batch size: 16 (per GPU) × 4 GPUs × 2 accum = 128

source ~/.bashrc
conda activate nvsubq

cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.
source .env

python experiments/run.py \
    --config examples/language_modeling/medium_hyena.py
