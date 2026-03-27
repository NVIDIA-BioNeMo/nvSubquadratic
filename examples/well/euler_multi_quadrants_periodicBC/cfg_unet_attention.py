"""UNet-Attention variant for euler_multi_quadrants_periodicBC.

Replaces ConvNeXt blocks with transformer blocks (multi-head self-attention
+ FFN).  Uses the same encoder-decoder skip-connection structure and
hyperparameters as cfg_unet_convnext.py.

WARNING: Full self-attention at 512x512 (262k tokens at the first encoder
stage) is extremely expensive.  This config is intended for architecture
comparison (FLOPs/params) and small-resolution testing, not production
training at full resolution.  For practical training consider reducing
spatial_resolution or using windowed attention.
"""

from examples.well.euler_multi_quadrants_periodicBC._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    SPATIAL_RESOLUTION,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet import AttentionBlock, WellUNet


# ─── Model hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE = 24
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 4
BLOCKS_AT_NECK = 1
NUM_HEADS = 6  # head_dim = init_features / num_heads = 7 at first stage
MLP_RATIO = 4
GRADIENT_CHECKPOINTING = False


def get_config() -> ExperimentConfig:
    """Build UNet-Attention experiment config for euler_multi_quadrants_periodicBC."""
    config = get_base_config(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    config.net = LazyConfig(WellUNet)(
        dim_in=IN_CHANNELS,
        dim_out=OUT_CHANNELS,
        n_spatial_dims=DATA_DIM,
        spatial_resolution=SPATIAL_RESOLUTION,
        stages=STAGES,
        blocks_per_stage=BLOCKS_PER_STAGE,
        blocks_at_neck=BLOCKS_AT_NECK,
        init_features=INIT_FEATURES,
        block_cfg=LazyConfig(AttentionBlock)(
            n_spatial_dims=DATA_DIM,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO,
        ),
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    return config
