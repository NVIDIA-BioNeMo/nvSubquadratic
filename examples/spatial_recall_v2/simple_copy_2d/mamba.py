# TODO: Add license header here

"""2D Simple Copy — Mamba XS (v2).

Hidden dim: 160, 4 blocks, headdim=32, expand=2, bidirectional.
~1.90M params (matching Attention ~1.84M / Hyena ~1.89M).
Bidirectional doubles Mamba2 layers, so hidden_dim is reduced from 208 → 160.

Note: hidden_dim must be a multiple of 16 for Mamba2.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.simple_copy_2d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 160
HEADDIM = 32
EXPAND = 2


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.optimizer.lr = 1e-3

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=True,
    )

    return config
