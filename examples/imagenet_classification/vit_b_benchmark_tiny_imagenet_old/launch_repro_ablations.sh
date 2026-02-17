#!/bin/bash
# Re-launch script for Phase 1 ablations that OOM'd
# This includes the OOM fix (wandb_checkpoint_upload=False in each config)

CONFIG_DIR="examples/imagenet_classification/vit_b_benchmark_tiny_imagenet"
LOG_DIR="slurm"

# 1. Baseline (Attention) - previously OOM'd at Epoch 252
sbatch --partition=geodude \
    --account=geodudeusers \
    --time=12:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=8 \
    --job-name=img_attn_v2 \
    --output=${LOG_DIR}/img_attn_v2_%j.out \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate nvsubq && export PYTHONPATH=. && export IMAGENETTE_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/imagenette' && export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf' && export HF_HUB_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub' && python experiments/run.py --config ${CONFIG_DIR}/imagenette_attention_patchify.py"

# 2. Split Weight Decay - previously OOM'd at Epoch 312
sbatch --partition=geodude \
    --account=geodudeusers \
    --time=12:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=8 \
    --job-name=img_split_wd_v2 \
    --output=${LOG_DIR}/img_split_wd_v2_%j.out \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate nvsubq && export PYTHONPATH=. && export IMAGENETTE_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/imagenette' && export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf' && export HF_HUB_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub' && python experiments/run.py --config ${CONFIG_DIR}/imagenette_hyena_split_wd.py"

# 3. High Frequency - previously OOM'd at Epoch 347
sbatch --partition=geodude \
    --account=geodudeusers \
    --time=12:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=8 \
    --job-name=img_hi_freq_v2 \
    --output=${LOG_DIR}/img_hi_freq_v2_%j.out \
    --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate nvsubq && export PYTHONPATH=. && export IMAGENETTE_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/imagenette' && export HF_HOME='/ivi/zfs/s0/original_homes/dwessel/data/.hf' && export HF_HUB_CACHE='/ivi/zfs/s0/original_homes/dwessel/data/.hf/hub' && python experiments/run.py --config ${CONFIG_DIR}/imagenette_hyena_omega_60.py"
