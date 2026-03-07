# TODO: Add license header here

"""EMNIST Spatial Recall 1D - Mamba XS Causal - Frequency/Memory Sweep Config.

This config is designed for sweeping Mamba's frequency/memory parameters, analogous to:
- L_cache in Hyena (controls kernel cache size / frequency grid)
- rope_base in Attention (controls RoPE frequency decay)

Key Mamba Parameters:
- A_init_range: Controls eigenvalues of A matrix (decay rate)
  - Smaller values (e.g., 0.1-1) = slower decay = longer memory
  - Larger values (e.g., 4-32) = faster decay = shorter memory
  - Default: (1, 16)

- dt_min/dt_max: Controls discretization time step initialization
  - Smaller values = finer temporal resolution = slower effective decay
  - Larger values = coarser temporal resolution = faster effective decay
  - Default: dt_min=0.001, dt_max=0.1

Sweep Plan (see TRACKER.md for details):
- A_init_range: (0.1, 1), (0.5, 4), (1, 16)
- dt: (0.0001, 0.001), (0.0001, 0.01), (0.001, 0.1)
- learning_rate: 1e-3, 1e-4, 1e-5

Model Size: XS (Extra-Small)
- Hidden dim: 128
- Headdim: 32
- Expand: 2
- d_state: 128
"""

import examples.spatial_recall_1d.mixer_defaults as spatial_recall_1d_mixer_defaults
from examples.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from examples.spatial_recall_1d.base_config import (
    base_experiment_config as spatial_recall_1d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
READOUT_VALUE = 0.0

# Network parameters - XS size
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 128
HEADDIM = 32
EXPAND = 2

# =============================================================================
# SWEEP PARAMETERS - Modify these for different runs
# =============================================================================

# Mamba frequency/memory parameters
D_STATE = 128  # State dimension (keep fixed)

# A_init_range options: (0.1, 1), (0.5, 4), (1, 16) [default]
A_INIT_RANGE = (0.5, 4)  # Slower decay than default

# dt options: (0.0001, 0.001), (0.0001, 0.01), (0.001, 0.1) [default]
DT_MIN = 0.0001
DT_MAX = 0.01

# Learning rate options: 1e-3, 1e-4 [default], 1e-5
LEARNING_RATE = 1e-4

# =============================================================================

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for Mamba frequency/memory sweep."""
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        learning_rate=LEARNING_RATE,
        wandb_job_group="spatial_recall_1d_mamba_freq_sweep",
    )

    # Mixer: Mamba2 with configurable memory settings
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=False,  # Causal!
        # Frequency/memory parameters
        d_state=D_STATE,
        A_init_range=A_INIT_RANGE,
        dt_min=DT_MIN,
        dt_max=DT_MAX,
    )

    # Dataset: 1D spatial recall
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_1d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        num_items=1,
        placement="fixed",
        with_mask=False,
        normalize_input=True,
        readout_value=READOUT_VALUE,
    )

    return config
