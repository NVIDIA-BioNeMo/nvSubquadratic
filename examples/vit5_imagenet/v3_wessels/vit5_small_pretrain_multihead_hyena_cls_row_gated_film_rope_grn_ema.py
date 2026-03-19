"""ViT-5-Small + Multi-Head Hyena ImageNet-1k — CLS-row, FiLM + RoPE + GRN.

Multi-head variant of the depthwise FiLM + RoPE + GRN config:
- CKConvMultiheadND (6 heads, head_dim=64) with dense within-head channel mixing.
- FiLM-conditioned SIREN kernels (input-dependent via register pooling).
- 2D RoPE on Q and K before gating.
- GRN after mixer output for inter-channel feature competition.
- Dual gating: SiLU (first gate) + Sigmoid (second gate).
- CLS-row architecture: CLS + 13 registers as extra row -> 15x14 grid.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    HIDDEN_DIM,
    build_cls_row_network,
    build_film_cfg,
    build_multihead_hyena_mixer,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.grn import GlobalResponseNorm


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Multi-Head Hyena CLS-row + FiLM + RoPE + GRN config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_multihead_hyena_mixer(film_cfg=film_cfg, use_rope=True)
    config.net, trainer_overrides = build_cls_row_network(
        mixer_cfg,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=HIDDEN_DIM),
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
