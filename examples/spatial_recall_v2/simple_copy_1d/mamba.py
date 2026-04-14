# TODO: Add license header here

"""1D Simple Copy — Mamba XS (v2, causal/unidirectional).

Hidden dim: 208, 4 blocks, headdim=32, expand=2.
~1.80M params (matching Attention ~1.84M / Hyena ~1.89M).
Unidirectional for causal 1D task.

Note: hidden_dim must be a multiple of 16 for Mamba2 (d_inner = expand*d_model
must be divisible by headdim=32).
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.simple_copy_1d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 208
HEADDIM = 32
EXPAND = 2


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.optimizer.lr = 5e-4
    config.train.grad_clip = 1.0

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=False,
    )

    return config
