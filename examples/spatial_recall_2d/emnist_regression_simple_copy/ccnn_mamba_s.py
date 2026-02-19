# TODO: Add license header here

"""EMNIST Spatial Recall 2D - Mamba S (Small).

Model Size: S (Small)
- Hidden dim: 160 (param-matched to Hyena/Attention S)
- Params: ~1.91M (bidirectional, expand=2, headdim=32)
- Heads: 10

Note: Mamba is very parameter-efficient, so it needs smaller hidden_dim
than Hyena/Attention to match total network params.

Size Reference:
- XS: ~700K-800K params
- S:  ~1.8M-2.0M params
"""

import examples.spatial_recall_2d.mixer_defaults as spatial_recall_2d_mixer_defaults
from examples.spatial_recall_2d.base_config import (
    base_emnist_spatial_recall_2d_dataset_config,
)
from examples.spatial_recall_2d.base_config import (
    base_experiment_config as spatial_recall_2d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16
CANVAS_SIZE = 64

# Network parameters - S size (param-matched to Hyena/Attention)
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 160  # Smaller than Hyena/Attn (256) due to Mamba efficiency
HEADDIM = 32  # inner_dim=320, heads=10
EXPAND = 2

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall with Mamba S."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_regression_simple_copy",
    )

    # Mixer: Mamba2 bidirectional (handles its own projections, NOT wrapped in QKVSequenceMixer)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=True,
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
