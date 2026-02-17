#!/bin/bash
# Launcher script for Phase 1 Ablations
# This script submits 3 separate jobs to SLURM.

CONFIG_DIR="examples/imagenet_classification/vit_b_benchmark_tiny_imagenet"
LOG_DIR="slurm"

# 1. Split Weight Decay
sbatch --partition=geodude \
    --account=geodudeusers \
    --time=12:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=8 \
    --job-name=img_split_wd \
    --output=${LOG_DIR}/img_split_wd_%j.out \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate nvsubq && export PYTHONPATH=. && export IMAGENETTE_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/imagenette' && export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf' && python experiments/run.py --config ${CONFIG_DIR}/imagenette_hyena_split_wd.py"

# 2. High Frequency
sbatch --partition=geodude \
    --account=geodudeusers \
    --time=12:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=8 \
    --job-name=img_hi_freq \
    --output=${LOG_DIR}/img_hi_freq_%j.out \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate nvsubq && export PYTHONPATH=. && export IMAGENETTE_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/imagenette' && export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf' && python experiments/run.py --config ${CONFIG_DIR}/imagenette_hyena_omega_60.py"

# 3. Deep Filter
sbatch --partition=geodude \
    --account=geodudeusers \
    --time=12:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=8 \
    --job-name=img_deep \
    --output=${LOG_DIR}/img_deep_%j.out \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate nvsubq && export PYTHONPATH=. && export IMAGENETTE_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/imagenette' && export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf' && python experiments/run.py --config ${CONFIG_DIR}/imagenette_hyena_deep_filter.py"
