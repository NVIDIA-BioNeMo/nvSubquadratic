# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CCNN diffusion on ImageNet 128x128, JiT-B matched, low omega_0 (3.0).

Identical to ccnn_jit_128 except omega_0 is reduced from 10.0 to 3.0 to
suppress high-frequency artifacts observed in earlier runs.
"""

import os

import torch

from experiments.datamodules._deprecated.ref_imagenet import ImageNetDataModule
from experiments.default_cfg import (
    DiffusionConfig,
    DiffusionExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import AdaLNZeroResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


WANDB_ENTITY = "dafidofff"

# Dataset ----------------------------------------------------------------------
BATCH_SIZE = 128
NUM_WORKERS = min(12, os.cpu_count() - 2 or 4)
FINAL_IMAGE_SIZE = 128
PATCH_SIZE = 8
HF_DATASET = os.environ.get("IMAGENET_HF_DATASET", "imagenet-1k")
HF_CACHE = os.environ.get("IMAGENET_PATH", "/scratch-shared/dknigge/hf_cache")

# Network params (CCNN matching JiT-B)
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
DATA_DIM = 2
NUM_HIDDEN_CHANNELS = 768
NUM_BLOCKS = 12
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "single"
FFT_PADDING = "circular"

# SIREN kernel — reduced omega_0 to suppress high-frequency artifacts
KERNEL_OMEGA_0 = 3.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# Optimisation -----------------------------------------------------------------
TRAINING_ITERATIONS = 250_000
WARMUP_ITERATIONS_PERCENTAGE = 0.02
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 2

# Diffusion --------------------------------------------------------------------
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 50
NUM_SAMPLES = 16
LOG_SAMPLES = True
EMA_ENABLED = True
EMA_DECAY = 0.9998
EMA_WARMUP_STEPS = 1000
EMA_UPDATE_EVERY = 1

# CFG
USE_CFG = True
GUIDANCE_SCALE = 2.9
CONDITION_DROPOUT_PROB = 0.1
NUM_CLASSES = 1000


def get_config() -> DiffusionExperimentConfig:
    """Build the experiment configuration."""
    config = DiffusionExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir=HF_CACHE,
        hf_dataset_name=HF_DATASET,
        image_size=FINAL_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=42,
        task="generation",
        drop_labels=False,
    )

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features="${net.in_channels}",
            out_features="${net.hidden_dim}",
            data_dim="${net.data_dim}",
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features="${net.hidden_dim}",
            out_features="${net.out_channels}",
            data_dim="${net.data_dim}",
            patch_size="${net.in_proj_cfg.patch_size}",
            stride="${net.in_proj_cfg.stride}",
            weight_init="zeros",
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(AdaLNZeroResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim="${net.data_dim}",
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim="${net.data_dim}",
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=32,
                            num_layers=3,
                            embedding_dim=32,
                            omega_0=KERNEL_OMEGA_0,
                            hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                            L_cache=FINAL_IMAGE_SIZE // PATCH_SIZE,
                            use_bias=True,
                        ),
                        mask_cfg=LazyConfig(torch.nn.Identity)(),
                        grid_type=GRID_TYPE,
                        fft_padding=FFT_PADDING,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels="3 * ${net.hidden_dim}",
                        out_channels="3 * ${net.hidden_dim}",
                        kernel_size=3,
                        groups="3 * ${net.hidden_dim}",
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.GroupNorm)(
                        num_groups=1,
                        num_channels="${net.hidden_dim}",
                    ),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                    use_rope=False,
                ),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
            condition_norm_cfg="${net.norm_cfg}",
            hidden_dim="${net.hidden_dim}",
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        condition_in_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features="${net.hidden_dim}", out_features="${net.hidden_dim}"
        ),
    )

    config.lightning_wrapper_class = LazyConfig(DiffusionWrapper)()

    config.optimizer = LazyConfig(torch.optim.Adam)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
    )

    config.scheduler = SchedulerConfig(
        name="constant",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.diffusion = DiffusionConfig(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        log_samples=LOG_SAMPLES,
        ema_enabled=EMA_ENABLED,
        ema_decay=EMA_DECAY,
        ema_update_every=EMA_UPDATE_EVERY,
        ema_warmup_steps=EMA_WARMUP_STEPS,
        use_classifier_free_guidance=USE_CFG,
        guidance_scale=GUIDANCE_SCALE,
        condition_dropout_prob=CONDITION_DROPOUT_PROB,
        num_classes=NUM_CLASSES,
        p_mean=-0.8,
        p_std=0.8,
        cfg_interval_start=0.1,
        cfg_interval_end=1.0,
        fid_online_jit=False,
        fid_stats_file="",
    )

    config.wandb = WandbConfig(
        job_group="imagenet_diffusion_ccnn_128_low_omega",
        entity=WANDB_ENTITY,
        tags=["ccnn", "diffusion", "imagenet128", "jit-B-matched", "low-omega"],
    )

    return config
