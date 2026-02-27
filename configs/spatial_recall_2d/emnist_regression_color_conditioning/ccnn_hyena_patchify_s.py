# TODO: Add license header here

"""EMNIST Spatial Recall 2D (Color Conditioning) - Hyena + Patchify S.

Model Size: S (Small)
- Hidden dim: 256

Task: 4 items on canvas with colored frames, output digit in matching color.
Input: 3-channel RGB with colored bounding boxes.
Output: 3-channel RGB digit colored with frame color.
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
NUM_ITEMS = 4

# Network parameters - S size
INPUT_CHANNELS = 3  # RGB with colored frames
OUTPUT_CHANNELS = 3  # RGB output (digit in frame color)
HIDDEN_DIM = 256

# Patchification parameters
PATCH_SIZE = 2
STRIDE = 2

# Training parameters
TRAINING_ITERATIONS = 50_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST color conditioning with Hyena + Patchify S."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_color_conditioning_s",
    )

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

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_hyena_mixer_cfg()

    # Fix L_cache for patchified sequence length
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.L_cache = (
        "${dataset.canvas_size} // ${net.in_proj_cfg.patch_size}"
    )

    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_2d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        use_colored_frames=True,  # RGB with colored bounding boxes
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=False,
        normalize_input=True,
        colored_label=True,  # Output colored digit (RGB)
    )

    return config
