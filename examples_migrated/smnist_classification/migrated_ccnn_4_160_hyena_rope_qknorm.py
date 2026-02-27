# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FULLY MIGRATED Sequential MNIST classification - ALL modules from nvsubq.

This is the complete production-style migration where ALL core modules come from nvsubq:
- HyenaConfig + Hyena (mixer)
- SIRENKernelND + SIRENKernelNDConfig (kernel)
- GaussianModulationND + GaussianModulationNDConfig (mask)
- MLPConfig + MLP (feedforward)
- Init functions (small_init, wang_init)

Only experiment infrastructure (ResidualBlock, ClassificationResNet, data modules) remain
in nvsubquadratic-private.

This config exactly matches test_old_ccnn_4_160_hyena_rope_qknorm.py parameters.

Key Features:
- RoPE (Rotary Position Embeddings): use_rope=True
- QK Normalization: apply_qk_norm=True
- SIREN kernel with Gaussian modulation
- 1D sequence processing (flattened MNIST)
"""

import os

import torch
from nvsubq import HyenaConfig, MLPConfig, QKVSequenceMixerConfig

from examples_migrated.mixer_factories import create_hyena_sequence_mixer, create_mlp
from experiments.datamodules.mnist import MNISTDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubq_paper.lazy_config import PLACEHOLDER, LazyConfig
from nvsubq_paper.modules.residual_block import ResidualBlock
from nvsubq_paper.networks.classification_resnet import ClassificationResNet


# Dataset parameters
INPUT_CHANNELS = 1  # MNIST grayscale
OUTPUT_CHANNELS = 10  # 10 digit classes
DATA_TYPE = "sequence"
DATA_DIM = 1

# Training parameters
BATCH_SIZE = 128
PRECISION = "bf16-mixed"  # Tested options: "32-true", "bf16-mixed"

# Model parameters
NUM_HIDDEN_CHANNELS = 160
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1

# TRAINING parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
GRAD_CLIP = 10.0
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001

# Hyena/SIREN kernel parameters (matches test_old exactly)
SHORT_CONV_KERNEL_SIZE = 3
KERNEL_TYPE = "siren"  # Use SIREN kernel
MASK_TYPE = "gaussian"  # Use Gaussian mask
GRID_TYPE = "double"
FFT_PADDING = "zero"
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 100.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
KERNEL_L_CACHE = 32

# MLP parameters
MLP_ACTIVATION = "glu"
MLP_EXPANSION_FACTOR = 1.0


def get_config() -> ExperimentConfig:
    """Return the FULLY MIGRATED Sequential MNIST classification configuration.

    This config uses:
    - HyenaConfig with SIREN kernel and Gaussian mask from nvsubq
    - MLPConfig from nvsubq
    - QKVSequenceMixerConfig from nvsubq
    - Init functions from nvsubq
    - RoPE (Rotary Position Embeddings)
    - QK Normalization

    It matches test_old_ccnn_4_160_hyena_rope_qknorm exactly but with production-style dataclass configs.
    """
    config = ExperimentConfig()

    # =========================================================================
    # Dataset Configuration (unchanged)
    # =========================================================================
    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        data_type=DATA_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        use_deterministic_worker_init=True,  # Flag to use deterministic worker initialization
        seed=config.seed,  # Pass the seed value instead of a Generator object
        task="classification",
    )

    # =========================================================================
    # FULLY MIGRATED: Hyena Configuration with SIREN + Gaussian
    # =========================================================================

    # Create HyenaConfig with SIREN kernel and Gaussian mask
    # This matches the test_old config exactly!
    hyena_config = HyenaConfig(
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        short_conv_kernel_size=SHORT_CONV_KERNEL_SIZE,
        is_causal=False,  # Not causal for MNIST
        use_pixelhyena_norm=True,
        use_output_norm=False,
        apply_qk_norm=True,  # QK Normalization enabled
        use_rope=True,  # RoPE enabled
        rope_base=10000.0,
        # Global conv config - SIREN kernel with Gaussian mask
        global_conv_kernel_type=KERNEL_TYPE,  # "siren"
        global_conv_mask_type=MASK_TYPE,  # "gaussian"
        global_conv_grid_type=GRID_TYPE,  # "double"
        global_conv_fft_padding=FFT_PADDING,  # "zero"
        global_conv_use_chunked_fftconv=False,
        # SIREN kernel parameters (matches test_old)
        kernel_mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,  # 32
        kernel_num_layers=KERNEL_NUM_LAYERS,  # 3
        kernel_embedding_dim=KERNEL_EMBEDDING_DIM,  # 32
        kernel_omega_0=KERNEL_OMEGA_0,  # 100.0
        kernel_hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,  # 1.0
        kernel_L_cache=KERNEL_L_CACHE,  # 32
        kernel_use_bias=True,
        # Gaussian mask parameters (matches test_old exactly)
        mask_min_std=0.025,
        mask_max_std=1.25,
        mask_init_std_low=0.05,
        mask_init_std_high=1.0,
        mask_parametrization="direct",
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
            # FULLY MIGRATED: Hyena with SIREN + Gaussian from nvsubq
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
            # Condition mixer
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            # Dropout
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
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
    )

    config.wandb = WandbConfig(
        job_group="smnist_classification_migrated",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    return config
