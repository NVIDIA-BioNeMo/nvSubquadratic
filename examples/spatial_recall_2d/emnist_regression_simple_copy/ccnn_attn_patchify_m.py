# TODO: Add license header here

"""EMNIST Spatial Recall 2D - Attention + Patchify M (Medium).

Model Size: M (Medium)
- Hidden dim: 384
- Params: ~4.4M + patchify overhead
- num_heads=12, head_dim=32 (consistent with S-size)

Patchification (ViT-style):
- Patchify: Conv2d with kernel_size=stride=patch_size (non-overlapping)
- Reduces sequence: 64x64=4096 → (64/patch_size)^2 tokens
- Unpatchify: ConvTranspose2d to reconstruct full resolution

Size Reference:
- XS: ~160 channels, 8 heads, head_dim=20
- S:  ~256 channels, 8 heads, head_dim=32
- M:  ~384 channels, 12 heads, head_dim=32
"""

import examples.spatial_recall_2d.mixer_defaults as spatial_recall_2d_mixer_defaults
from examples.spatial_recall_2d.base_config import (
    base_emnist_spatial_recall_2d_dataset_config,
)
from examples.spatial_recall_2d.base_config import (
    base_experiment_config as spatial_recall_2d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.patchify import Patchify, Unpatchify


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16
CANVAS_SIZE = 64

# Network parameters - M size
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 384
NUM_HEADS = 12  # head_dim = 384/12 = 32 (consistent with S-size)

# Patchification parameters (default, can be overridden via CLI)
PATCH_SIZE = 8  # 64/8 = 8x8 patches = 64 tokens
STRIDE = 8  # Non-overlapping patches (ViT-style)

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall with Attention + Patchify M.

    With PATCH_SIZE=8 on a 64x64 canvas:
    - Input: [B, 64, 64, 1] → Patchify → [B, 8, 8, 384] (64 tokens vs 4096)
    - After blocks: [B, 8, 8, 384] → Unpatchify → [B, 64, 64, 1]
    - Readout: [B, 64, 64, 1] → [B, 16, 16, 1]
    """
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_regression_simple_copy",
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
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    # Mixer: Multi-head self-attention
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
    )

    # Dataset
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_2d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        use_colored_frames=False,
        num_items=1,
        placement="fixed",
        with_mask=False,
        normalize_input=True,
    )

    return config
