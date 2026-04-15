# TODO: Add license header here

"""2D Color Conditioning — Hyena XS with Gaussian mask + register-based FiLM (v2).

Same as hyena_gaussian_mask_film.py but uses **register tokens** (ViT-5
pattern) instead of global-average-pool self-conditioning.  The network
prepends a row of 4 learnable register embeddings along the first spatial
dimension.  Each block extracts and pools these registers (via
``RegisterPooling``) to produce a ``[B, C]`` conditioning vector for the
``KernelFiLMGenerator``, making the SIREN kernel input-dependent.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.color_conditioning_2d._base import base_experiment_config
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.masks_nd import GaussianModulationND


HIDDEN_DIM = 256
DATA_DIM = 2
NUM_FILM_LAYERS = 2  # matches KERNEL_NUM_LAYERS - 1 hidden linears
NUM_REGISTERS = 4


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_compatible_fftconv = True
    config.optimizer.lr = 5e-4
    config.train.grad_clip = 1.0

    # Register tokens on the network (prepends a register row)
    config.net.num_registers = NUM_REGISTERS

    # Register-based FiLM conditioning on the block
    config.net.block_cfg.register_pooling_cfg = LazyConfig(RegisterPooling)(num_registers=NUM_REGISTERS)
    config.net.block_cfg.num_registers = NUM_REGISTERS

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_hyena_mixer_cfg(
        short_conv_cfg=mixer_defaults.short_conv_cfg(data_dim=2),
        film_cfg=LazyConfig(KernelFiLMGenerator)(
            cond_dim=HIDDEN_DIM,
            kernel_hidden_dim=mixer_defaults.KERNEL_MLP_HIDDEN_DIM,
            num_film_layers=NUM_FILM_LAYERS,
            film_hidden_dim=64,
        ),
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
