"""Hyena config with Gaussian modulation mask for euler_multi_quadrants_periodicBC.

Identical to ``cfg_hyena.py`` but replaces the ``nn.Identity`` mask with a
``GaussianModulationND`` mask on the CKConv global convolution kernel.
"""

from examples.well.euler_multi_quadrants_periodicBC.cfg_hyena import (
    DATA_DIM,
    NUM_HIDDEN_CHANNELS,
)
from examples.well.euler_multi_quadrants_periodicBC.cfg_hyena import (
    get_config as _get_hyena_config,
)
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


def get_config() -> ExperimentConfig:
    """Build Hyena + Gaussian mask config for euler_multi_quadrants_periodicBC."""
    config = _get_hyena_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(GaussianModulationND)(
        data_dim=DATA_DIM,
        num_channels=NUM_HIDDEN_CHANNELS,
        min_std=0.025,
        max_std=1.25,
        init_std_low=0.05,
        init_std_high=1.0,
        parametrization="direct",
    )

    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))

    return config
