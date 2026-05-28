# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""JiT-L/16 on ImageNet 256x256 — exact JiT replication.

The README does not ship a JiT-L training script, but the JiT-L
*evaluation* script uses ``--cfg 2.4`` for JiT-L/16, which implies the
released JiT-L/16 model was trained with the same hyperparameters as
JiT-B/16 (the recipe is shared across sizes) and only the eval CFG differs.

Equivalent reference command::

    torchrun --nproc_per_node=8 main_jit.py \\
        --model JiT-L/16 \\
        --proj_dropout 0.0 \\
        --P_mean -0.8 --P_std 0.8 \\
        --img_size 256 --noise_scale 1.0 \\
        --batch_size 128 --blr 5e-5 \\
        --epochs 600 --warmup_epochs 5 \\
        --gen_bsz 128 --num_images 50000 --cfg 2.4 \\
        --interval_min 0.1 --interval_max 1.0 \\
        --online_eval

Effective batch = 128 / GPU x 8 GPUs = 1024 (LR = blr * 1024 / 256 = 2e-4).
"""

from examples.imagenet_diffusion_jit._base_config import get_base_config
from experiments.default_cfg import DiffusionExperimentConfig
from nvsubquadratic.networks.jit import JiT_L_16


# ─── JiT-L/16 @ 256x256 ─────────────────────────────────────────────────────
MODEL_FACTORY = JiT_L_16
IMAGE_SIZE = 256
# Reference: 128/GPU x 8 GPUs x 1 = 1024 effective batch.
# L is ~3x larger than B (around 460M params); on 8x H100/H200 (80 GB)
# the 128/GPU batch fits in bf16-mixed.  If you run on smaller GPUs,
# halve ``BATCH_SIZE`` and double ``ACCUMULATE_GRAD_STEPS``.
BATCH_SIZE = 128
NUM_GPUS = 8
ACCUMULATE_GRAD_STEPS = 1

NOISE_SCALE = 1.0
GUIDANCE_SCALE = 2.4  # README eval cfg for JiT-L/16 @ 256
PROJ_DROPOUT = 0.0


def get_config() -> DiffusionExperimentConfig:
    """Build the JiT-L/16 256x256 experiment configuration."""
    return get_base_config(
        model_factory=MODEL_FACTORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_gpus=NUM_GPUS,
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
        noise_scale=NOISE_SCALE,
        guidance_scale=GUIDANCE_SCALE,
        proj_dropout=PROJ_DROPOUT,
        job_group="imagenet_diffusion_jit_l16_256",
        extra_tags=["JiT-L/16", "256x256"],
    )
