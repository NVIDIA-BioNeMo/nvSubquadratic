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

import os
import torch

from experiments.datamodules.imagenet import ImageNetDataModule
from experiments.default_cfg import (
    DiffusionConfig,
    DiffusionExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers import DiffusionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.huggingface_diffusers import DiffusersUVitWrapper, HuggingFaceUVitConfig


PLACEHOLDER = None

# Dataset ----------------------------------------------------------------------
BATCH_SIZE = 32
MAX_WORKERS = 8
NUM_WORKERS = min(MAX_WORKERS, os.cpu_count() or MAX_WORKERS)
IMAGE_SIZE = 256
FINAL_IMAGE_SIZE = 64
IMAGENET_CACHE_DIR = os.environ.get("IMAGENET_CACHE", "/projects/0/prjs1161/imagenet")
HF_DATASET_NAME = "imagenet-1k"
HF_DATASET_CONFIG = None

# UVit architecture ------------------------------------------------------------
UVIT_SAMPLE_SIZE = FINAL_IMAGE_SIZE
UVIT_IN_CHANNELS = 3
UVIT_OUT_CHANNELS = 3
UVIT_HIDDEN_SIZE = 256
UVIT_COND_EMBED_DIM = 128
UVIT_ENCODER_HIDDEN_SIZE = 128
UVIT_BLOCK_OUT_CHANNELS = 256
UVIT_NUM_HIDDEN_LAYERS = 8
UVIT_NUM_ATTENTION_HEADS = 8
UVIT_INTERMEDIATE_SIZE = 512
UVIT_LAYER_NORM_EPS = 1e-5
UVIT_MICRO_COND_ENCODE_DIM = None
UVIT_MICRO_COND_EMBED_DIM = None
UVIT_CODEBOOK_SIZE = None
UVIT_VOCAB_SIZE = None

# Optimisation -----------------------------------------------------------------
TRAINING_ITERATIONS = 800_000
WARMUP_ITERATIONS_PERCENTAGE = 0.02
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Diffusion --------------------------------------------------------------------
NUM_TRAIN_TIMESTEPS = 1_000
NUM_INFERENCE_STEPS = 50
NUM_SAMPLES = 8
LOG_SAMPLES = True
EMA_ENABLED = True
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 1_000
EMA_UPDATE_EVERY = 1


def get_config() -> DiffusionExperimentConfig:
    """Build the experiment configuration."""
    config = DiffusionExperimentConfig()
    config.debug = False
    config.seed = 42

    hf_token = os.environ.get("HF_TOKEN")

    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir=IMAGENET_CACHE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=FINAL_IMAGE_SIZE,
        center_crop=True,
        drop_labels=True,
        hf_dataset_name=HF_DATASET_NAME,
        hf_dataset_config=HF_DATASET_CONFIG,
        hf_auth_token=hf_token,
    )

    hf_cfg = HuggingFaceUVitConfig(
        sample_size=UVIT_SAMPLE_SIZE,
        in_channels=UVIT_IN_CHANNELS,
        out_channels=UVIT_OUT_CHANNELS,
        hidden_size=UVIT_HIDDEN_SIZE,
        cond_embed_dim=UVIT_COND_EMBED_DIM,
        encoder_hidden_size=UVIT_ENCODER_HIDDEN_SIZE,
        block_out_channels=UVIT_BLOCK_OUT_CHANNELS,
        num_hidden_layers=UVIT_NUM_HIDDEN_LAYERS,
        num_attention_heads=UVIT_NUM_ATTENTION_HEADS,
        intermediate_size=UVIT_INTERMEDIATE_SIZE,
        layer_norm_eps=UVIT_LAYER_NORM_EPS,
        micro_cond_encode_dim=UVIT_MICRO_COND_ENCODE_DIM,
        micro_cond_embed_dim=UVIT_MICRO_COND_EMBED_DIM,
        codebook_size=UVIT_CODEBOOK_SIZE,
        vocab_size=UVIT_VOCAB_SIZE,
    )

    config.net = LazyConfig(DiffusersUVitWrapper)(hf_config=hf_cfg)
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
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
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
    )

    config.wandb = WandbConfig(job_group="imagenet_diffusion_hf_uvit_baseline")

    return config
