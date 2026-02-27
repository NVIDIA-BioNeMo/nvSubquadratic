# TODO: Add license header here

"""EMNIST Spatial Recall 3D - Hyena XS (Extra-Small).

3D Spatial Recall Task:
- 2D images placed on depth slices of a 3D volume [D, H, W]
- Target placed at front-top-left (fixed) or random position
- Must recall target at back-bottom-right corner (last depth slice)

Model Size: XS (Extra-Small)
- Hidden dim: 160
- Params: ~767K (similar to 2D version)

Size Reference:
- XS: ~160 channels (~700K-1M params)
- S:  ~256 channels (~1.8M-2.2M params)
"""

import configs.spatial_recall_3d.mixer_defaults as spatial_recall_3d_mixer_defaults
from configs.spatial_recall_3d.base_config import (
    base_emnist_spatial_recall_3d_dataset_config,
)
from configs.spatial_recall_3d.base_config import (
    base_experiment_config as spatial_recall_3d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16
CANVAS_SIZE = 64  # H and W dimensions
CANVAS_DEPTH = 8  # D dimension

# Network parameters - XS size
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 160

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 3D with Hyena XS."""
    config = spatial_recall_3d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_3d_emnist_regression_simple_copy",
        target_size=TARGET_SIZE,
    )

    # Mixer: Hyena with SIREN kernel
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_3d_mixer_defaults.get_hyena_mixer_cfg()

    # Dataset
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_3d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        canvas_depth=CANVAS_DEPTH,
        batch_size=BATCH_SIZE,
        num_items=1,
        placement="fixed",
        with_mask=False,
        normalize_input=True,
    )

    return config
