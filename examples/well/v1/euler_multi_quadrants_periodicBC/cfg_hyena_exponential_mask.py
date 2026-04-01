"""Hyena config with Exponential modulation mask for euler_multi_quadrants_periodicBC.

Identical to ``cfg_hyena.py`` but replaces the ``nn.Identity`` mask with an
``ExponentialModulationND`` mask on the CKConv global convolution kernel.
"""

from examples.well.euler_multi_quadrants_periodicBC.cfg_hyena import (
    DATA_DIM,
    NUM_HIDDEN_CHANNELS,
)
from examples.well.euler_multi_quadrants_periodicBC.cfg_hyena import (
    get_config as _get_hyena_config,
)
from experiments.callbacks.iteration_speed import IterationSpeedCallback
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import ExponentialModulationND


def get_config() -> ExperimentConfig:
    """Build Hyena + Exponential mask config for euler_multi_quadrants_periodicBC."""
    config = _get_hyena_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(ExponentialModulationND)(
        data_dim=DATA_DIM,
        num_channels=NUM_HIDDEN_CHANNELS,
        fast_decay_pct=13.81,
        slow_decay_pct=2.3,
    )

    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(LazyConfig(IterationSpeedCallback)(log_every_n_steps=10))

    return config
