# TODO: Add license header here

"""EMNIST Spatial Recall 1D (Mask Selection) - Mamba M Causal + Patchify.

1D version of mask selection with patchification:
- Input: [B, 4096, 2] → Patchify (patch_size=64) → [B, 64, hidden_dim]
- Process: Mamba on 64 tokens (vs 4096 without patchification)
- Output: [B, 64, hidden_dim] → Unpatchify → [B, 4096, 1]

Task:
- 4 flattened images placed in 1D canvas (4096 elements)
- Binary mask channel indicates which digit to recall
- Model must regress the target region for the masked digit (causal)

Model Size: M (Medium) - ~5.0M params
- Hidden dim: 352
- Headdim: 32
- Expand: 2
"""

import configs.spatial_recall_1d.mixer_defaults as spatial_recall_1d_mixer_defaults
from configs.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from configs.spatial_recall_1d.base_config import (
    base_experiment_config as spatial_recall_1d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER, LazyConfig
from nvsubq_paper.modules.patchify import Patchify, Unpatchify


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
NUM_ITEMS = 4  # 1 target + 3 distractors

# Patchification parameters
PATCH_SIZE = 64  # 4096 / 64 = 64 tokens
STRIDE = PATCH_SIZE  # Non-overlapping patches

# Network parameters - M size (causal Mamba allows larger hidden_dim)
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 352  # Causal M (~5M params target)
HEADDIM = 32
EXPAND = 2

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST mask selection 1D with Mamba M + Patchify."""
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_1d_emnist_mask_selection_m_patchify",
    )

    # Override in/out projections with Patchify/Unpatchify
    config.net.in_proj_cfg = LazyConfig(Patchify)(
        in_features="${net.in_channels}",
        out_features="${net.hidden_dim}",
        data_dim="${net.data_dim}",
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )
    config.net.out_proj_cfg = LazyConfig(Unpatchify)(
        in_features="${net.hidden_dim}",
        out_features="${net.out_channels}",
        data_dim="${net.data_dim}",
        patch_size="${net.in_proj_cfg.patch_size}",
        stride="${net.in_proj_cfg.stride}",
    )

    # Mixer: Mamba2 (unidirectional/causal)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=False,  # Causal!
    )

    # Dataset: 1D spatial recall with mask selection
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_1d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=True,
        normalize_input=True,
    )

    return config
