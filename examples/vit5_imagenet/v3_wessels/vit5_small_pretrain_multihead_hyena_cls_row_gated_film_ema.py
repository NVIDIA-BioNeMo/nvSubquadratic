"""ViT-5-Small + Multi-Head Hyena ImageNet-1k — CLS-row, FiLM-conditioned SIREN.

Multi-head variant of the depthwise FiLM config:
- CKConvMultiheadND (6 heads, head_dim=64) with dense within-head channel mixing.
- FiLM-conditioned SIREN kernels (input-dependent via register pooling).
- Dual gating: SiLU (first gate) + Sigmoid (second gate).
- CLS-row architecture: CLS + 13 registers as extra row -> 15x14 grid.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    build_cls_row_network,
    build_film_cfg,
    build_multihead_hyena_mixer,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Multi-Head Hyena CLS-row + FiLM config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_multihead_hyena_mixer(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(mixer_cfg)
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
