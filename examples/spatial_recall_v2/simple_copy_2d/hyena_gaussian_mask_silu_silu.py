# TODO: Add license header here

"""2D Simple Copy — Hyena XS with Gaussian mask, SiLU/SiLU gating ablation.

Same as hyena_gaussian_mask.py but both gates use SiLU (instead of SiLU + Sigmoid).
"""

import torch

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.simple_copy_2d._base import base_experiment_config
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


HIDDEN_DIM = 256
DATA_DIM = 2


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_compatible_fftconv = True
    config.optimizer.lr = 5e-4
    config.train.grad_clip = 1.0

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_hyena_mixer_cfg(
        short_conv_cfg=mixer_defaults.short_conv_cfg(2),
        gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
        gate_nonlinear_2_cfg=LazyConfig(torch.nn.SiLU)(),
    )

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
