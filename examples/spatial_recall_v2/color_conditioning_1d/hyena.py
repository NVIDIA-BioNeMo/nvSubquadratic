# TODO: Add license header here

"""1D Color Conditioning — Hyena XS (v2, causal).

Hidden dim: 256, 4 blocks.
4 items on 1D canvas with coloured boundary markers, output digit in matching colour.
Causal by default; pass is_causal=False for bidirectional variant.
Compile: max-autotune-no-cudagraphs + compile_compatible_fftconv.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.color_conditioning_1d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 256


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_compatible_fftconv = True
    config.optimizer.lr = 5e-4
    config.train.grad_clip = 1.0

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_hyena_mixer_cfg(
        short_conv_cfg=mixer_defaults.short_conv_cfg(data_dim=1),
        is_causal=True,
        fft_backend="torch_fft",
    )

    return config
