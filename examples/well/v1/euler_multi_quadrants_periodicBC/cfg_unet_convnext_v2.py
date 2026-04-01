"""CNextU-net V2 for euler_multi_quadrants_periodicBC.

Same hyperparameters as cfg_unet_convnext.py (run yc5ncpxa) but uses
UNetConvNextV2 which fixes the skips[0] bug — all encoder skip connections
are used in the decoder, including the finest-resolution features.
"""

from examples.well.v1.euler_multi_quadrants_periodicBC._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    SPATIAL_RESOLUTION,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet_convnext_v2 import WellUNetConvNextV2


BATCH_SIZE = 24
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 4
BLOCKS_AT_NECK = 1
GRADIENT_CHECKPOINTING = False


def get_config() -> ExperimentConfig:
    """Build CNextU-net V2 experiment config for euler_multi_quadrants_periodicBC."""
    config = get_base_config(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    config.net = LazyConfig(WellUNetConvNextV2)(
        dim_in=IN_CHANNELS,
        dim_out=OUT_CHANNELS,
        n_spatial_dims=DATA_DIM,
        spatial_resolution=SPATIAL_RESOLUTION,
        stages=STAGES,
        blocks_per_stage=BLOCKS_PER_STAGE,
        blocks_at_neck=BLOCKS_AT_NECK,
        init_features=INIT_FEATURES,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    return config
