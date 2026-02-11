# TODO: Add license header here

"""EMNIST Spatial Recall 2D (Color Conditioning) - Hyena Multi-Head M.

Model Size: M (Medium)
- Hidden dim: 384 (divisible by num_heads for clean head_dim)
- Num heads: 48 (head_dim = 384 / 48 = 8)
- head_dim: 8
- Params: ~5M

This variant uses multi-head convolutions where each head has dense
[head_dim x head_dim] channel mixing, similar to multi-head attention.

Comparison to standard Hyena:
- Standard: depthwise conv, no channel mixing within conv
- Multi-head: dense conv within each head, cross-channel learning

Task: 4 items on canvas with colored frames, output digit in matching color.
Input: 3-channel RGB with colored bounding boxes.
Output: 3-channel RGB digit colored with frame color.
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

# Network parameters - M size with multi-head
INPUT_CHANNELS = 3  # RGB with colored frames
OUTPUT_CHANNELS = 3  # RGB output (digit in frame color)
# HIDDEN_DIM = 352  # Increased for ~5M params, divisible by num_heads
# NUM_HEADS = 11 # 48  # head_dim = 384 / 48 = 8

HIDDEN_DIM = 384
NUM_HEADS = 24


# Training parameters
TRAINING_ITERATIONS = 50_000
CHECKPOINT_EVERY_N_STEPS = 2000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST color conditioning with Hyena Multi-Head M."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_color_conditioning_hyena_multihead_m",
    )

    # Mixer: Multi-Head Hyena with SIREN kernel
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_hyena_multihead_mixer_cfg(
        num_heads=NUM_HEADS,
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
        colored_label=True,  # Output colored digit (RGB)
    )

    # Checkpointing every 2000 steps to avoid losing progress on preemption
    config.train.checkpoint_every_n_steps = CHECKPOINT_EVERY_N_STEPS

    # Add LayerStatsCallback for debugging activation/gradient norms
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
