# TODO: Add license header here

"""EMNIST Spatial Recall 2D - Hyena + Patchify XS (Extra-Small).

Model Size: XS (Extra-Small)
- Hidden dim: 160
- Params: ~0.7-0.8M

Patchification (ViT-style):
- Patchify: Conv2d with kernel_size=stride=8 (non-overlapping)
- Reduces sequence: 64x64=4096 → 8x8=64 tokens
- Unpatchify: ConvTranspose2d to reconstruct full resolution

This tests whether Hyena benefits from (or is hurt by) shorter sequences.
"""

import configs.spatial_recall_2d.mixer_defaults as spatial_recall_2d_mixer_defaults
from configs.spatial_recall_2d.base_config import (
    base_emnist_spatial_recall_2d_dataset_config,
)
from configs.spatial_recall_2d.base_config import (
    base_experiment_config as spatial_recall_2d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER, LazyConfig
from nvsubq_paper.modules.patchify import Patchify, Unpatchify


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16
CANVAS_SIZE = 64

# Network parameters - XS size
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 160

# Patchification parameters
PATCH_SIZE = 8  # 64/8 = 8x8 patches = 64 tokens
STRIDE = 8  # Non-overlapping patches (ViT-style)

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall with Hyena + Patchify XS.

    With PATCH_SIZE=8 on a 64x64 canvas:
    - Input: [B, 64, 64, 1] → Patchify → [B, 8, 8, 160] (64 tokens vs 4096)
    - After blocks: [B, 8, 8, 160] → Unpatchify → [B, 64, 64, 1]
    """
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_simple_copy_xs",
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

    # Mixer: Hyena with SIREN kernel
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_hyena_mixer_cfg()

    # Fix L_cache for patchified sequence length (canvas_size // patch_size)
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.L_cache = (
        "${dataset.canvas_size} // ${net.in_proj_cfg.patch_size}"
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
