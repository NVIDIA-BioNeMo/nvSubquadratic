# TODO: Add license header here

"""2D Color Conditioning — Hyena XS with Patchify + Gaussian mask (v2).

Identical to ``hyena_patch.py`` but replaces the ``nn.Identity`` mask with a
``GaussianModulationND`` mask on the CKConv global convolution kernel.

Patch-size CLI override
-----------------------
Only ``net.in_proj_cfg.patch_size=P`` is needed; stride, out_proj, and
L_cache are derived via interpolators.
"""

from examples.spatial_recall_v2.color_conditioning_2d.hyena_patch import HIDDEN_DIM
from examples.spatial_recall_v2.color_conditioning_2d.hyena_patch import get_config as _get_hyena_patch_config
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


DATA_DIM = 2


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = _get_hyena_patch_config()

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
