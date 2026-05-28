# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# TODO: Add license header here

"""Config file for ImageNet diffusion using the shared ResNet backbone."""

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
from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import AdaLNZeroResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# Dataset parameters
INPUT_CHANNELS = 3  # RGB images
OUTPUT_CHANNELS = 3  # Reconstruct RGB
NUM_CLASSES = 1_000
DATA_DIM = 2

# Training parameters
BATCH_SIZE = 42
IMAGENET_PATH = os.environ.get("IMAGENET_CACHE", "/home/dknigge/project_dir/huggingface/imagenet")
HF_DATASET_NAME = "imagenet-1k"
HF_DATASET_CONFIG = None
IMAGE_SIZE = 256
FINAL_IMAGE_SIZE = 64
PRECISION = "bf16-mixed"  # Tested options: "32-true", "bf16-mixed"

# Model parameters
NUM_HIDDEN_CHANNELS = 768
NUM_BLOCKS = 12
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "single"
FFT_PADDING = "circular"

# Optimisation parameters
TRAINING_ITERATIONS = 800_000
WARMUP_ITERATIONS_PERCENTAGE = 0.02
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 2e-4
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 1

# Diffusion parameters
PREDICTION_TYPE = "v_prediction"
NUM_TRAIN_TIMESTEPS = 1_000
BETA_START = 1e-4
BETA_END = 2e-2
BETA_SCHEDULE = "cosine_interpolated"
TIME_EMBED_DIM = NUM_HIDDEN_CHANNELS
MAX_PERIOD = 10_000.0
LOG_SAMPLES = True

# Classifier-free guidance
CFG_ENABLED = True
GUIDANCE_SCALE = 3.5
CONDITION_DROPOUT_PROB = 0.25


def get_config() -> DiffusionExperimentConfig:
    """Return the ImageNet diffusion configuration."""
    config = DiffusionExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True

    hf_token = os.environ.get("HF_TOKEN")

    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir=IMAGENET_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=FINAL_IMAGE_SIZE,
        center_crop=True,
        drop_labels=False,
        hf_dataset_name="imagenet-1k",
        hf_dataset_config=None,
        hf_auth_token=hf_token,
        task="generation",
    )

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(AdaLNZeroResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim="${net.data_dim}",
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(RandomFourierKernelND)(
                            data_dim="${net.data_dim}",
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=64,
                            num_layers=3,
                            embedding_dim=64,
                            omega_0=100.0,
                            L_cache=32,
                            use_bias=True,
                            nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim="${net.data_dim}",
                            num_channels="${net.hidden_dim}",
                            min_attenuation_at_step=0.1,
                            max_attenuation_at_limit=0.95,
                            init_extent=1.0,
                            parametrization="direct",
                        ),
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

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
        precision="bf16-mixed",
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.diffusion = DiffusionConfig(
        prediction_type=PREDICTION_TYPE,
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule=BETA_SCHEDULE,
        cosine_schedule_image_resolution=FINAL_IMAGE_SIZE,
        cosine_schedule_noise_res_high=FINAL_IMAGE_SIZE,
        cosine_schedule_noise_res_low=max(32, FINAL_IMAGE_SIZE // 2),
        time_embed_dim=TIME_EMBED_DIM,
        max_period=MAX_PERIOD,
        num_classes=1_000,
        use_classifier_free_guidance=CFG_ENABLED,
        guidance_scale=GUIDANCE_SCALE,
        condition_dropout_prob=CONDITION_DROPOUT_PROB,
        fid_enabled=False,
    )

    config.wandb = WandbConfig(
        job_group="imagenet-diffusion",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    return config
