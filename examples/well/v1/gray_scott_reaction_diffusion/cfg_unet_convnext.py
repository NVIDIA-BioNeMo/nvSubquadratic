"""CNextU-net baseline for gray_scott_reaction_diffusion.

Reference: Well paper Table 6 — best LR for CNextU-net = 1e-4 (15 epochs in 12h).
Table 2: VRMSE = 0.1761.

At 128×128 (16× fewer pixels than Euler 512×512), batch_size can be much
larger.  We use bf16-mixed + torch.compile for higher throughput.
"""

from examples.well.v1.gray_scott_reaction_diffusion._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    SPATIAL_RESOLUTION,
    get_base_config,
)
from experiments.callbacks.iteration_speed import IterationSpeedCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet_convnext import WellUNetConvNext


# ─── Model hyperparameters (matching Well paper Appendix E.1) ─────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  # Table 6: best LR for CNextU-net on gray_scott
WEIGHT_DECAY = 1e-4

INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 4
BLOCKS_AT_NECK = 1
GRADIENT_CHECKPOINTING = False


def get_config() -> ExperimentConfig:
    """Build CNextU-net experiment config for gray_scott_reaction_diffusion."""
    config = get_base_config(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

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

    config.callbacks.append(LazyConfig(IterationSpeedCallback)(log_every_n_steps=10))

    return config
