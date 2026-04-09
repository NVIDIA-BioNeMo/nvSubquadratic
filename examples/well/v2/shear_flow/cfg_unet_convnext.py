"""CNextU-net baseline for shear_flow (v2).

CNextU-net best LR: 5e-4 (paper Table 6).
"""

from examples.well.v2.shear_flow._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    SPATIAL_RESOLUTION,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet_convnext import WellUNetConvNext


INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 4
BLOCKS_AT_NECK = 1
GRADIENT_CHECKPOINTING = False


def get_config() -> ExperimentConfig:
    """Build CNextU-net experiment config."""
    config = get_base_config()

    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    config.net = LazyConfig(WellUNetConvNext)(
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
