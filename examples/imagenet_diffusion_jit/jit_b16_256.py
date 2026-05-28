# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""JiT-B/16 on ImageNet 256x256 — exact JiT replication.

Direct port of the JiT-B/16 256x256 training script from
``~/projects/JiT/README.md``::

    torchrun --nproc_per_node=8 main_jit.py \\
        --model JiT-B/16 \\
        --proj_dropout 0.0 \\
        --P_mean -0.8 --P_std 0.8 \\
        --img_size 256 --noise_scale 1.0 \\
        --batch_size 128 --blr 5e-5 \\
        --epochs 600 --warmup_epochs 5 \\
        --gen_bsz 128 --num_images 50000 --cfg 2.9 \\
        --interval_min 0.1 --interval_max 1.0 \\
        --online_eval

Effective batch = 128 / GPU x 8 GPUs = 1024 (LR = blr * 1024 / 256 = 2e-4).
Use ``accumulate_grad_steps`` to recover the same effective batch on fewer GPUs.
"""

from examples.imagenet_diffusion_jit._base_config import get_base_config
from experiments.default_cfg import DiffusionExperimentConfig
from nvsubquadratic.networks.jit import JiT_B_16


# ─── JiT-B/16 @ 256x256, paper config ────────────────────────────────────────
MODEL_FACTORY = JiT_B_16
IMAGE_SIZE = 256
# Reference: 128/GPU x 8 GPUs x 1 = 1024 effective batch.
BATCH_SIZE = 128
NUM_GPUS = 8
ACCUMULATE_GRAD_STEPS = 1

NOISE_SCALE = 1.0  # README: --noise_scale 1.0 for 256x256
GUIDANCE_SCALE = 2.9  # README: --cfg 2.9 for JiT-B/16 training
PROJ_DROPOUT = 0.0  # README: --proj_dropout 0.0 for B/L; 0.2 for H


def get_config() -> DiffusionExperimentConfig:
    """Build the JiT-B/16 256x256 experiment configuration."""
    return get_base_config(
        model_factory=MODEL_FACTORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_gpus=NUM_GPUS,
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
        noise_scale=NOISE_SCALE,
        guidance_scale=GUIDANCE_SCALE,
        proj_dropout=PROJ_DROPOUT,
        job_group="imagenet_diffusion_jit_b16_256",
        extra_tags=["JiT-B/16", "256x256"],
    )
