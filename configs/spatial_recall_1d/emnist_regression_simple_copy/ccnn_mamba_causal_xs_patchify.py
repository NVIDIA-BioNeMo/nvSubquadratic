# TODO: Add license header here

"""EMNIST Spatial Recall 1D - Mamba XS Causal + Patchify.

Patchified version of Mamba for 1D spatial recall:
- Input: [B, 4096, 1] → Patchify (patch_size=64) → [B, 64, hidden_dim]
- Process: Mamba on 64 tokens (vs 4096 without patchification)
- Output: [B, 64, hidden_dim] → Unpatchify → [B, 4096, 1]

Benefits:
- 64x shorter sequence length (4096 → 64 tokens)
- Captures local structure within patches
- More efficient for long sequences

Model Size: XS (Extra-Small)
- Hidden dim: 128
- Headdim: 32
- Expand: 2
"""

import configs.spatial_recall_1d.mixer_defaults as spatial_recall_1d_mixer_defaults
from configs.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from configs.spatial_recall_1d.base_config import (
    base_experiment_config as spatial_recall_1d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER, LazyConfig
from nvsubq_paper.modules.patchify import Patchify, Unpatchify


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
HIDDEN_DIM = 128
HEADDIM = 32
EXPAND = 2

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 1D with Mamba XS + Patchify.

    With PATCH_SIZE=64 on a 4096-element canvas:
    - Input: [B, 4096, 1] → Patchify → [B, 64, 128] (64 tokens vs 4096)
    - After blocks: [B, 64, 128] → Unpatchify → [B, 4096, 1]
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

    # Mixer: Mamba2 (causal)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=False,  # Causal!
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
