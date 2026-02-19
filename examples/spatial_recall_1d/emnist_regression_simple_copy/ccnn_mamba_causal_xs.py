# TODO: Add license header here

"""EMNIST Spatial Recall 1D - Mamba XS (Extra-Small).

This is the 1D version of the spatial recall task where:
1. Images are flattened FIRST (16×16 → 256 elements)
2. Flattened image placed as contiguous segment in 1D canvas (4096 elements)
3. Model must recall the flattened image

Mamba is designed for 1D sequences:
- Native 1D sequence processing
- Linear complexity O(n) vs O(n²) for attention
- Bidirectional mode for non-causal tasks
- May struggle with "find and copy" tasks due to lack of explicit position attention

Model Size: XS (Extra-Small)
- Hidden dim: 128
- Headdim: 32
- Expand: 2
- Params: ~738K (unidirectional)
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
# For unidirectional Mamba (bidirectional=False), we need ~738K params to match Attention's ~719K
# hidden_dim must be multiple of 16 for Mamba2 (d_ssm % headdim == 0)
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 128  # Gives ~738K params for unidirectional Mamba / 96 for bidirectional Mamba
HEADDIM = 32
EXPAND = 2

# Training parameters
TRAINING_ITERATIONS = 20_000  # ~2 epochs @ BS=64


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 1D with Mamba XS."""
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_1d_emnist_simple_copy_xs",
    )

    # Mixer: Mamba2 (bidirectional for non-causal task)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=False,  # Causal!
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
