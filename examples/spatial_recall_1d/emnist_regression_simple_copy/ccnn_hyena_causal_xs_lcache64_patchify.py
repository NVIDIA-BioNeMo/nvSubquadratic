# TODO: Add license header here

"""EMNIST Spatial Recall 1D - Hyena XS Causal + Patchify (L_cache=64).

Patchified version of Hyena for 1D spatial recall:
- Input: [B, 4096, 1] → Patchify (patch_size=64) → [B, 64, hidden_dim]
- Process: Hyena with L_cache=64 on 64 tokens (kernel covers full sequence)
- Output: [B, 64, hidden_dim] → Unpatchify → [B, 4096, 1]

L_cache=64 matches the patchified sequence length:
- SIREN kernel generates a 64-element convolution kernel
- This kernel covers the entire patchified sequence
- Equivalent to having full receptive field after patchification

Benefits:
- 64x shorter sequence length (4096 → 64 tokens)
- Kernel size matches sequence length (full receptive field)
- More efficient FFT convolution on shorter sequences

Model Size: XS (Extra-Small)
- Hidden dim: 160
- SIREN kernel with 3 layers
- L_cache: 64 (matches patchified sequence length)
"""

import examples.spatial_recall_1d.mixer_defaults as spatial_recall_1d_mixer_defaults
from examples.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from examples.spatial_recall_1d.base_config import (
    base_experiment_config as spatial_recall_1d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.utils.qk_norm import L2Norm
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.utils.qk_norm import L2Norm


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
READOUT_VALUE = 0.0

# Patchification parameters
PATCH_SIZE = 64  # 4096 / 64 = 64 tokens
STRIDE = PATCH_SIZE  # Non-overlapping patches

# Network parameters - XS size
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 160

# Hyena-specific: L_cache matches patchified sequence length
L_CACHE = 64  # = canvas_length / patch_size = 4096 / 64

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 1D with Hyena XS + Patchify.

    With PATCH_SIZE=64 on a 4096-element canvas:
    - Input: [B, 4096, 1] → Patchify → [B, 64, 160] (64 tokens vs 4096)
    - L_cache=64 means the SIREN kernel covers the full patchified sequence
    - After blocks: [B, 64, 160] → Unpatchify → [B, 4096, 1]
    """
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_1d_emnist_simple_copy_xs_patchify",
    )

    # Override in/out projections with Patchify/Unpatchify
    config.net.in_proj_cfg = LazyConfig(Patchify)(
        in_features="${net.in_channels}",
        out_features="${net.hidden_dim}",
        data_dim="${net.data_dim}",
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )
    config.net.out_proj_cfg = LazyConfig(Unpatchify)(
        in_features="${net.hidden_dim}",
        out_features="${net.out_channels}",
        data_dim="${net.data_dim}",
        patch_size="${net.in_proj_cfg.patch_size}",
        stride="${net.in_proj_cfg.stride}",
    )

    # Mixer: Hyena with causal convolutions, L_cache=64
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_hyena_mixer_cfg(
        is_causal=True,  # Causal mode!
        qk_norm_cfg=LazyConfig(L2Norm)(),
        L_cache="${dataset.canvas_size}",  # 64 instead of 4096!
    )

    # Dataset: 1D spatial recall
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_1d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        num_items=1,
        placement="fixed",
        with_mask=False,
        normalize_input=True,
        readout_value=READOUT_VALUE,
    )

    return config
