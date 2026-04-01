"""UNet-Hyena variant for euler_multi_quadrants_periodicBC.

Replaces ConvNeXt blocks with Hyena blocks (gated global convolution via
CKConvND with SIREN kernels + FFN).  Uses the same encoder-decoder
skip-connection structure and hyperparameters as cfg_unet_convnext.py.

The Hyena global convolution uses circular FFT padding, which is natural
for this dataset's periodic boundary conditions.  The SIREN kernel's
L_cache is automatically set per stage to match the spatial resolution
after downsampling.
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
from nvsubquadratic.networks.baselines.unet import HyenaBlock, WellUNet


# ─── Model hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE = 24
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 4
BLOCKS_AT_NECK = 1
MLP_RATIO = 4
GRADIENT_CHECKPOINTING = False

# Hyena / SIREN parameters
OMEGA_0 = 30.0
SIREN_LAYERS = 3
SIREN_HIDDEN_DIM = 64


def get_config() -> ExperimentConfig:
    """Build UNet-Hyena experiment config for euler_multi_quadrants_periodicBC."""
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
        block_cfg=LazyConfig(HyenaBlock)(
            n_spatial_dims=DATA_DIM,
            mlp_ratio=MLP_RATIO,
            omega_0=OMEGA_0,
            siren_layers=SIREN_LAYERS,
            siren_hidden_dim=SIREN_HIDDEN_DIM,
        ),
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    return config
