"""Hyena + Gaussian mask config for acoustic_scattering_maze (v2) with circular FFT padding.

Identical to ``hyena_gaussian_mask.py`` but uses circular (periodic) FFT padding
instead of zeros, to compare periodic vs. non-periodic convolution on this dataset.
"""

from examples.well.v2.acoustic_scattering_maze.hyena_gaussian_mask import get_config as _get_hyena_gaussian_mask_config
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Build Hyena + Gaussian mask + circular padding config for acoustic_scattering_maze."""
    config = _get_hyena_gaussian_mask_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.fft_padding = "circular"

    return config
