"""Hyena config with Gaussian modulation mask for active_matter (v2).

Identical to ``cfg_hyena.py`` but replaces the ``nn.Identity`` mask with a
``GaussianModulationND`` mask on the CKConv global convolution kernel.
"""

from examples.well.v2.active_matter._base import DATA_DIM
from examples.well.v2.active_matter.hyena import NUM_HIDDEN_CHANNELS
from examples.well.v2.active_matter.hyena import get_config as _get_hyena_config

# from experiments.callbacks.mask_monitor import MaskMonitorCallback  # disabled: wandb artifact API retry loop blocks training
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


def get_config() -> ExperimentConfig:
    """Build Hyena + Gaussian mask config for active_matter."""
    config = _get_hyena_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(GaussianModulationND)(
        data_dim=DATA_DIM,
        num_channels=NUM_HIDDEN_CHANNELS,
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=1.0,
        parametrization="direct",
    )

    # config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))  # disabled: wandb artifact API retry loop blocks training

    return config
