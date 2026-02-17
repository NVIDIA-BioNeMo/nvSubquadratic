# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MIGRATED EMNIST Spatial Recall 2D (Mask Selection) - Hyena XS.

Alternative version using LazyConfig with an importable factory function.
"""

from nvsubq import HyenaConfig, QKVSequenceMixerConfig

from examples.spatial_recall_2d.base_config import (
    base_emnist_spatial_recall_2d_dataset_config,
)
from examples.spatial_recall_2d.base_config import (
    base_experiment_config as spatial_recall_2d_base_experiment_config,
)
from examples_migrated.mixer_factories import create_hyena_sequence_mixer
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


# Dataset-specific parameters
BATCH_SIZE = 32
TARGET_SIZE = 16
CANVAS_SIZE = 64
NUM_ITEMS = 4

# Network parameters - XS size
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 1
HIDDEN_DIM = 160
NUM_BLOCKS = 4

# Training parameters
TRAINING_ITERATIONS = 20_000

# Hyena / SIREN kernel parameters
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
SHORT_CONV_KERNEL_SIZE = 3
SHORT_CONV_ACCELERATED = True


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 2D mask selection with Hyena XS (migrated)."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_mask_selection_xs_migrated",
    )

    hyena_config = HyenaConfig(
        hidden_dim=HIDDEN_DIM,
        data_dim=2,
        short_conv_kernel_size=SHORT_CONV_KERNEL_SIZE,
        is_causal=False,
        short_conv_accelerated=SHORT_CONV_ACCELERATED,
        use_pixelhyena_norm=True,
        use_output_norm=False,
        apply_qk_norm=True,
        use_rope=False,
        rope_base=10000.0,
        global_conv_kernel_type="siren",
        global_conv_grid_type="double",
        global_conv_fft_padding="zero",
        global_conv_use_chunked_fftconv=False,
        kernel_mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        kernel_num_layers=KERNEL_NUM_LAYERS,
        kernel_embedding_dim=KERNEL_EMBEDDING_DIM,
        kernel_omega_0=KERNEL_OMEGA_0,
        kernel_hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
        kernel_L_cache=CANVAS_SIZE,
    )

    qkv_config = QKVSequenceMixerConfig(
        hidden_dim=HIDDEN_DIM,
        init_method_in="small",
        init_method_out="wang",
        num_layers=NUM_BLOCKS,
        bias=False,
    )

    # Use LazyConfig with importable factory function
    config.net.block_cfg.sequence_mixer_cfg = LazyConfig(create_hyena_sequence_mixer)(
        hyena_config=hyena_config,
        qkv_config=qkv_config,
    )

    config.dataset = base_emnist_spatial_recall_2d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        use_colored_frames=False,
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=True,
        normalize_input=True,
    )

    return config
