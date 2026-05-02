"""Full Mamba – 12 bidirectional Mamba blocks.

Layout: M M M M M M M M M M M M

Replaces every sequence mixer with a ``Mamba`` (mamba_ssm core, bidirectional)
block.  Everything else — training recipe, MLP, norms, drop-path, patch size,
CLS token, registers — is identical to ``full_hyena.py`` and ``full_attention.py``.

Mamba hyperparameters (mamba_ssm defaults):
  d_state=16, d_conv=4, expand=2, bidirectional=True

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


LAYER_PATTERN = "M" * NUM_BLOCKS


def get_config() -> ExperimentConfig:
    """Return the full-Mamba (12 M) hybrid config."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    config.wandb.job_group = "vit5_hybrid"
    return config
