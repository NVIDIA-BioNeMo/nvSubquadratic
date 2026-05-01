# TODO: Add license header here

"""2D Simple Copy — Hyena XS with Gaussian mask (v2).

Identical to ``hyena.py`` but replaces the ``nn.Identity`` mask with a
``GaussianModulationND`` mask on the CKConv global convolution kernel.
"""

from examples.spatial_recall_v2.simple_copy_2d.hyena import HIDDEN_DIM
from examples.spatial_recall_v2.simple_copy_2d.hyena import get_config as _get_hyena_config
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


DATA_DIM = 2


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = _get_hyena_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(GaussianModulationND)(
        data_dim=DATA_DIM,
        num_channels=HIDDEN_DIM,
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=1.0,
        parametrization="direct",
    )

    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))

    return config
