# TODO: Add license header here

"""EMNIST Spatial Recall 3D - Mask Selection - Mamba XS (Extra-Small).

3D Spatial Recall Task with Mask Selection:
- Multiple 2D images placed on depth slices of a 3D volume [D, H, W]
- One target, multiple distractors (all on different depth slices or positions)
- Mask channel indicates which item is the target
- Must recall target at back-bottom-right corner (last depth slice)

Model Size: XS (Extra-Small)
- Hidden dim: 96 (smaller to match param count)
- Params: ~0.78M (similar to 2D version)

Size Reference:
- XS: ~96 channels (~0.78M params for Mamba)
- S:  ~160 channels (~1.8M-2.2M params)
"""

import examples.spatial_recall_3d.mixer_defaults as spatial_recall_3d_mixer_defaults
from examples.spatial_recall_3d.base_config import (
    base_emnist_spatial_recall_3d_dataset_config,
)
from examples.spatial_recall_3d.base_config import (
    base_experiment_config as spatial_recall_3d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 16
TARGET_SIZE = 16
CANVAS_SIZE = 64  # H and W dimensions
CANVAS_DEPTH = 8  # D dimension

# Network parameters - XS size (smaller hidden for Mamba to match params)
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 96  # Smaller to match ~0.78M params

NUM_ITEMS = 4  # target + 3 distractors

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 3D mask selection with Mamba XS."""
    config = spatial_recall_3d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_3d_emnist_mask_selection_xs",
        target_size=TARGET_SIZE,
    )

    # Mixer: Mamba (bidirectional)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_3d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=32,
        bidirectional=True,
    )

    # Dataset
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_3d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        canvas_depth=CANVAS_DEPTH,
        batch_size=BATCH_SIZE,
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=True,
        normalize_input=True,
    )

    return config
