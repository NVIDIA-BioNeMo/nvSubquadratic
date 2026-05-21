# TODO: Add license header here

"""3D Color Conditioning -- Hyena XS with Gaussian mask (BASELINE for 2x2).

Ablation corner:
    - ``init_extent = 1.0`` (scalar, isotropic)
    - ``omega_0 = KERNEL_OMEGA_0`` (default = 10)

Paired with:
    - ``hyena_gaussian_mask_peraxis.py``  (per-axis init only)
    - ``hyena_gaussian_mask_omega30.py``  (omega_0 = 30 only)
    - ``hyena_gaussian_mask.py``          (per-axis init + omega_0 = 30)

Together these form a 2x2 design isolating the effect of per-axis
``init_extent`` vs. bumping the SIREN's ``omega_0`` on the anisotropic
[D=8, H=64, W=64] color-conditioning volume.
"""

from examples.spatial_recall_v2.color_conditioning_3d.hyena import HIDDEN_DIM
from examples.spatial_recall_v2.color_conditioning_3d.hyena import get_config as _get_hyena_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


DATA_DIM = 3


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

    return config
