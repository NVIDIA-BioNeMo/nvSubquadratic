# TODO: Add license header here

"""2D Simple Copy — Mamba XS (v2), expand=1 ablation.

Hidden dim: 224, 4 blocks, headdim=32, expand=1, bidirectional.
~1.70M params (vs ~1.60M with hidden_dim=160, expand=2).
hidden_dim increased from 160 to 224 to compensate for halved expansion.

Note: d_inner = expand * d_model must be divisible by headdim=32.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.simple_copy_2d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 224
HEADDIM = 32
EXPAND = 1


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
