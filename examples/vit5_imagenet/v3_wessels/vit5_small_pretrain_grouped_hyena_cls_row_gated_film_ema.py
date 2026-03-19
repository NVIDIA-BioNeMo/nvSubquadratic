"""ViT-5-Small + Grouped Hyena ImageNet-1k — CLS-row, FiLM-conditioned SIREN.

Grouped depthwise FiLM config (Evo-2-inspired weight sharing):
- CKConvND with grouped depthwise convolution (num_groups filters shared across hidden_dim).
- FiLM-conditioned SIREN kernels (input-dependent via register pooling).
- Dual gating: SiLU (first gate) + Sigmoid (second gate).
- CLS-row architecture: CLS + 13 registers as extra row -> 15x14 grid.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    build_cls_row_network,
    build_film_cfg,
    build_grouped_hyena_mixer,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Grouped Hyena CLS-row + FiLM config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_grouped_hyena_mixer(num_groups=6, film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(mixer_cfg)
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
