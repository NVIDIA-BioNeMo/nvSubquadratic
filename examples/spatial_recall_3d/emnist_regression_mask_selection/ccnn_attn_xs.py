# TODO: Add license header here

"""EMNIST Spatial Recall 3D - Mask Selection - Attention XS (Extra-Small).

3D Spatial Recall Task with Mask Selection:
- Multiple 2D images placed on depth slices of a 3D volume [D, H, W]
- One target, multiple distractors (all on different depth slices or positions)
- Mask channel indicates which item is the target
- Must recall target at back-bottom-right corner (last depth slice)

Model Size: XS (Extra-Small)
- Hidden dim: 160
- Params: ~0.72M (similar to 2D version)

Size Reference:
- XS: ~160 channels (~0.72M params for Attention)
- S:  ~256 channels (~1.8M-2.2M params)
"""

import examples.spatial_recall_3d.mixer_defaults as spatial_recall_3d_mixer_defaults
from examples.spatial_recall_3d.base_config import (
    base_emnist_spatial_recall_3d_dataset_config,
)
from examples.spatial_recall_3d.base_config import (
    base_experiment_config as spatial_recall_3d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 16
TARGET_SIZE = 16
CANVAS_SIZE = 64  # H and W dimensions
CANVAS_DEPTH = 8  # D dimension

# Network parameters - XS size
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 160
NUM_HEADS = 8  # head_dim = 160/8 = 20

NUM_ITEMS = 4  # target + 3 distractors

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 3D mask selection with Attention XS."""
    config = spatial_recall_3d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_3d_emnist_mask_selection_xs",
        target_size=TARGET_SIZE,
    )

    # Mixer: Attention (no RoPE for 3D - head_dim must be divisible by 6)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_3d_mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        use_rope=False,  # Disable RoPE for 3D (head_dim=20 not divisible by 6)
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
