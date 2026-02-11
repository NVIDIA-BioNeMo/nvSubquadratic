# TODO: Add license header here

"""EMNIST Spatial Recall 2D (Color Conditioning) - Hyena Multi-Head M with GroupNorm.

Same as ccnn_hyena_multihead_m.py but uses GroupNorm (per-head normalization)
instead of LayerNorm for pixelhyena_norm.
"""

import examples.spatial_recall_2d.mixer_defaults as spatial_recall_2d_mixer_defaults
from examples.spatial_recall_2d.base_config import (
    base_emnist_spatial_recall_2d_dataset_config,
)
from examples.spatial_recall_2d.base_config import (
    base_experiment_config as spatial_recall_2d_base_experiment_config,
)
from experiments.callbacks.layer_stats import LayerStatsCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16
CANVAS_SIZE = 64
NUM_ITEMS = 4

# Network parameters
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
HIDDEN_DIM = 352  # Divisible by 11 heads
NUM_HEADS = 11  # head_dim = 32

# Training parameters
TRAINING_ITERATIONS = 50_000
CHECKPOINT_EVERY_N_STEPS = 2000


def get_config() -> ExperimentConfig:
    """Get config with GroupNorm (per-head normalization)."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_color_conditioning_hyena_multihead_m_groupnorm",
    )

    # Mixer: Multi-Head Hyena with GroupNorm (per-head normalization)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_hyena_multihead_mixer_cfg(
        num_heads=NUM_HEADS,
        norm_mode="groupnorm",  # Per-head normalization
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

    config.train.checkpoint_every_n_steps = CHECKPOINT_EVERY_N_STEPS

    # Add LayerStatsCallback
    config.callbacks.append(
        LazyConfig(LayerStatsCallback)(
            log_every_n_steps=100,
            log_activations=True,
            log_gradients=True,
            track_residual_blocks=True,
            track_ckconv_layers=True,
        )
    )

    return config
