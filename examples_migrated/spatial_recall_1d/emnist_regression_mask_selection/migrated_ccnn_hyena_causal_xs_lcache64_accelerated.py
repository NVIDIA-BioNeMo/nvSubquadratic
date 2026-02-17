# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EMNIST Spatial Recall 1D (Mask Selection) - Hyena XS Causal with L_cache=64 (FULLY MIGRATED).

This is the fully migrated version using dataclass configs for all modules.


Model Size: XS (Extra-Small)
- Hidden dim: 160
- SIREN kernel with 3 layers
- L_cache: 64 (NOT 4096!)

Migration Status:
ALL MODULES MIGRATED - QKVSequenceMixer, Hyena, CKConvND, SIRENKernelND all use dataclass configs
"""

from nvsubq import HyenaConfig, QKVSequenceMixerConfig

from examples.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from examples.spatial_recall_1d.base_config import (
    base_experiment_config as spatial_recall_1d_base_experiment_config,
)
from examples_migrated.mixer_factories import create_hyena_sequence_mixer
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
NUM_ITEMS = 4  # 1 target + 3 distractors

# Network parameters - XS size
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 160
NUM_BLOCKS = 4  # From base_experiment_config default

# Training parameters
TRAINING_ITERATIONS = 20_000

# Hyena/CKConv kernel parameters
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
SHORT_CONV_KERNEL_SIZE = 3
SHORT_CONV_ACCELERATED = True


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST mask selection 1D with Hyena XS (L_cache=64).

    This is the MIGRATED version using QKVSequenceMixerConfig.
    """
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_1d_emnist_mask_selection_xs_migrated",
    )

    # =========================================================================
    # FULLY MIGRATED: All modules use dataclass configs
    # =========================================================================

    # Create HyenaConfig with nested CKConvND and SIREN parameters
    hyena_config = HyenaConfig(
        hidden_dim=HIDDEN_DIM,
        data_dim=1,  # 1D sequence
        short_conv_kernel_size=SHORT_CONV_KERNEL_SIZE,
        is_causal=True,  # Causal mode!
        short_conv_accelerated=SHORT_CONV_ACCELERATED,
        use_pixelhyena_norm=True,
        use_output_norm=False,
        apply_qk_norm=True,
        use_rope=False,
        rope_base=10000.0,
        # Global conv config (CKConvND parameters)
        global_conv_kernel_type="siren",
        global_conv_grid_type="double",
        global_conv_fft_padding="zero",
        global_conv_use_chunked_fftconv=False,
        # SIREN kernel config
        kernel_mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        kernel_num_layers=KERNEL_NUM_LAYERS,
        kernel_embedding_dim=KERNEL_EMBEDDING_DIM,
        kernel_omega_0=KERNEL_OMEGA_0,
        kernel_hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
        kernel_L_cache=CANVAS_SIZE,
    )

    # Create QKVSequenceMixerConfig
    qkv_config = QKVSequenceMixerConfig(
        hidden_dim=HIDDEN_DIM,
        init_method_in="small",
        init_method_out="wang",
        num_layers=NUM_BLOCKS,  # Required for wang_init
        bias=False,
    )

    # Use LazyConfig with importable factory function (otherwise we will only get one mixer instance for all blocks)
    config.net.block_cfg.sequence_mixer_cfg = LazyConfig(create_hyena_sequence_mixer)(
        hyena_config=hyena_config,
        qkv_config=qkv_config,
    )

    # =========================================================================
    # Dataset Configuration (unchanged)
    # =========================================================================

    config.dataset = base_emnist_spatial_recall_1d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        num_items=NUM_ITEMS,  # 4 items: 1 target + 3 distractors
        placement="random",  # Random placement
        with_mask=True,  # Binary mask channel to indicate target
        normalize_input=True,
    )

    return config
