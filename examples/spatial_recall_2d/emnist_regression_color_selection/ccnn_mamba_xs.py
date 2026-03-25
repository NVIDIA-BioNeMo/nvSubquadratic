# TODO: Add license header here

"""EMNIST Spatial Recall 2D (Color Selection) - Mamba XS (Extra-Small).

Model Size: XS (Extra-Small)
- Hidden dim: 96 (smaller to match param count)
- Params: ~0.78M

Task: 4 items on canvas with colored frames, select digit by color.
Input: 3-channel RGB with colored bounding boxes.
Note: Running XS to verify if Mamba can learn color-based selection.
"""

import examples.spatial_recall_2d.mixer_defaults as spatial_recall_2d_mixer_defaults
from examples.spatial_recall_2d.base_config import (
    base_emnist_spatial_recall_2d_dataset_config,
)
from examples.spatial_recall_2d.base_config import (
    base_experiment_config as spatial_recall_2d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16
CANVAS_SIZE = 64
NUM_ITEMS = 4  # 1 target + 3 distractors

# Network parameters - XS size (smaller hidden for Mamba to match params)
INPUT_CHANNELS = 3  # RGB with colored frames
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 96  # Smaller to match ~0.78M params

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST color selection with Mamba XS."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_color_selection_xs",
    )

    # Mixer: Mamba (bidirectional)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=32,
        bidirectional=True,
    )

    # Dataset
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_2d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        use_colored_frames=True,  # RGB with colored bounding boxes
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=False,  # No mask, selection by color
        normalize_input=True,
    )

    return config
