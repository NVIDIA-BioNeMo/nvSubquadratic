# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FULLY MIGRATED TinyImageNet classification - ALL modules from nvsubq.

This is the complete production-style migration where ALL core modules come from nvsubq:
- HyenaConfig + Hyena (mixer)
- RandomFourierKernelND + RandomFourierKernelNDConfig (kernel - matches test_old exactly)
- GaussianModulationND + GaussianModulationNDConfig (mask - matches test_old exactly)
- MLPConfig + MLP (feedforward)
- Init functions (small_init, wang_init)

Only experiment infrastructure (ResidualBlock, ClassificationResNet, data modules) remain
in nvsubquadratic-private.

This config exactly matches test_old_supertiny_ccnn_7_512_hyena_circular.py parameters.
"""

import os

import torch
from nvsubq import HyenaConfig, MLPConfig, QKVSequenceMixerConfig

from examples_migrated.mixer_factories import create_hyena_sequence_mixer, create_mlp
from experiments.datamodules.tinyimagenet import TinyImageNetDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.classification_resnet import ClassificationResNet


# Dataset parameters
INPUT_CHANNELS = 3  # RGB images
OUTPUT_CHANNELS = NUM_CLASSES = 200  # TinyImageNet classes
DATA_DIM = 2

# Training parameters
BATCH_SIZE = 32
IMAGENET_PATH = os.environ.get("TINYIMAGENET_CACHE", os.path.expanduser("~/.cache/tinyimagenet"))
HF_DATASET_NAME = "zh-plus/tiny-imagenet"
HF_DATASET_CONFIG = None
IMAGE_SIZE = 64
FINAL_IMAGE_SIZE = 64
PRECISION = "bf16-mixed"

# Model parameters (SUPER-TINY: 4 blocks)
NUM_HIDDEN_CHANNELS = 512
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1

# Optimisation parameters
TRAINING_ITERATIONS = 600_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

# Hyena/RandomFourier kernel parameters (matches test_old exactly)
SHORT_CONV_KERNEL_SIZE = 3
KERNEL_TYPE = "random_fourier"  # Use RandomFourier (not SIREN) to match test_old
MASK_TYPE = "gaussian"  # Use Gaussian mask to match test_old
GRID_TYPE = "single"
FFT_PADDING = "circular"
KERNEL_MLP_HIDDEN_DIM = 64
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 64
KERNEL_OMEGA_0 = 100.0
KERNEL_L_CACHE = 32

# MLP parameters
MLP_ACTIVATION = "glu"
MLP_EXPANSION_FACTOR = 2.0


def get_config() -> ExperimentConfig:
    """Return the FULLY MIGRATED TinyImageNet classification configuration.

    This config uses:
    - HyenaConfig with RandomFourier kernel and Gaussian mask from nvsubq
    - MLPConfig from nvsubq
    - QKVSequenceMixerConfig from nvsubq
    - Init functions from nvsubq

    It matches test_old_supertiny exactly but with production-style dataclass configs.
    """
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    hf_token = os.environ.get("HF_TOKEN")

    # =========================================================================
    # Dataset Configuration (unchanged)
    # =========================================================================
    config.dataset = LazyConfig(TinyImageNetDataModule)(
        data_dir=IMAGENET_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=FINAL_IMAGE_SIZE,
        center_crop=False,  # TinyImageNet is already 64x64
        num_classes=NUM_CLASSES,
        drop_labels=False,
        hf_dataset_name=HF_DATASET_NAME,
        hf_dataset_config=HF_DATASET_CONFIG,
        hf_auth_token=hf_token,
        task="classification",
    )

    # =========================================================================
    # FULLY MIGRATED: Hyena Configuration with RandomFourier + Gaussian
    # =========================================================================

    # Create HyenaConfig with RandomFourier kernel and Gaussian mask
    # This matches the test_old config exactly!
    hyena_config = HyenaConfig(
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        short_conv_kernel_size=SHORT_CONV_KERNEL_SIZE,
        is_causal=False,  # Not causal for images
        use_pixelhyena_norm=True,
        pixelhyena_norm_type="group",  # Use GroupNorm to match test_old
        pixelhyena_norm_num_groups=1,  # num_groups=1 (LayerNorm-like behavior)
        use_output_norm=False,
        apply_qk_norm=True,
        use_rope=True,
        rope_base=10000.0,
        # Global conv config - RandomFourier kernel with Gaussian mask
        global_conv_kernel_type=KERNEL_TYPE,  # "random_fourier"
        global_conv_mask_type=MASK_TYPE,  # "gaussian"
        global_conv_grid_type=GRID_TYPE,  # "single"
        global_conv_fft_padding=FFT_PADDING,  # "circular"
        global_conv_use_chunked_fftconv=False,
        # RandomFourier kernel parameters (matches test_old)
        kernel_mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,  # 64
        kernel_num_layers=KERNEL_NUM_LAYERS,  # 3
        kernel_embedding_dim=KERNEL_EMBEDDING_DIM,  # 64
        kernel_omega_0=KERNEL_OMEGA_0,  # 100.0
        kernel_L_cache=KERNEL_L_CACHE,  # 32
        kernel_use_bias=True,
    )

    # Create QKVSequenceMixerConfig
    qkv_config = QKVSequenceMixerConfig(
        hidden_dim=NUM_HIDDEN_CHANNELS,
        init_method_in="small",
        init_method_out="wang",
        num_layers=NUM_BLOCKS,
        bias=False,
    )

    # =========================================================================
    # FULLY MIGRATED: MLP Configuration
    # =========================================================================

    # Create MLPConfig from nvsubq
    mlp_config = MLPConfig(
        dim=NUM_HIDDEN_CHANNELS,
        activation=MLP_ACTIVATION,
        expansion_factor=MLP_EXPANSION_FACTOR,
        dropout_rate=DROPOUT_RATE,
        bias=False,
        init_method_in="small",
        init_method_out="wang",
        num_layers=NUM_BLOCKS,
    )

    # =========================================================================
    # Network Configuration (uses LazyConfig for experiment infrastructure)
    # =========================================================================
    config.net = LazyConfig(ClassificationResNet)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            # FULLY MIGRATED: Hyena with RandomFourier + Gaussian from nvsubq
            sequence_mixer_cfg=LazyConfig(create_hyena_sequence_mixer)(
                hyena_config=hyena_config,
                qkv_config=qkv_config,
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            # FULLY MIGRATED: MLP from nvsubq
            mlp_cfg=LazyConfig(create_mlp)(
                mlp_config=mlp_config,
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    # =========================================================================
    # Training Configuration (unchanged)
    # =========================================================================
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    config.wandb = WandbConfig(
        job_group="tinyimagenet_classification_fully_migrated",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    return config
