# TODO: Add license header here

"""EMNIST Spatial Recall 1D (Mask Selection) - Attention S Causal.

1D version of mask selection where:
1. Images are flattened FIRST (16×16 → 256 elements)
2. 4 flattened images placed as contiguous segments in 1D canvas (4096 elements)
3. Binary mask channel indicates which digit to recall
4. Model must regress the target region for the masked digit (causal)

Model Size: S (Small)
- Hidden dim: 256
- Num heads: 8 (head_dim=32)
- Causal attention with RoPE
"""

import configs.spatial_recall_1d.mixer_defaults as spatial_recall_1d_mixer_defaults
from configs.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from configs.spatial_recall_1d.base_config import (
    base_experiment_config as spatial_recall_1d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubq_paper.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
NUM_ITEMS = 4  # 1 target + 3 distractors

# Network parameters - S size
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 256
NUM_HEADS = 8  # head_dim = 256 / 8 = 32

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST mask selection 1D with Attention S (causal)."""
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_1d_emnist_mask_selection_s",
    )

    # Mixer: Multi-head self-attention (causal)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
        is_causal=True,  # Causal!
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
