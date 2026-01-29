# TODO: Add license header here

"""EMNIST Spatial Recall 2D (Color Conditioning) - Delta-Hyena M (No Patchify).

Model Size: M (Medium)
- Hidden dim: 416

Task: 4 items on canvas with colored frames, output digit in matching color.
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
NUM_ITEMS = 4

# Network parameters - M size
INPUT_CHANNELS = 3  # RGB with colored frames
OUTPUT_CHANNELS = 3  # RGB output (digit in frame color)
HIDDEN_DIM = 416

# Training parameters
TRAINING_ITERATIONS = 50_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST color conditioning with Delta-Hyena M (no patchify)."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_color_conditioning_delta_m",
    )

    # Mixer: Associative Delta-Hyena
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_delta_hyena_mixer_cfg(
        num_heads=8,
        gamma_init=0.1,
    )

    # Dataset
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_2d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        use_colored_frames=True,
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=False,
        normalize_input=True,
        colored_label=True,
    )

    return config
