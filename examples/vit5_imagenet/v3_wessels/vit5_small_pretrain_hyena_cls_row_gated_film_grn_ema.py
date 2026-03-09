"""ViT-5-Small + Hyena ImageNet-1k — CLS-row, FiLM + GRN (no RoPE).

Depthwise FiLM + GRN config:
- CKConvND (depthwise) with global channel mixing.
- FiLM-conditioned SIREN kernels (input-dependent via register pooling).
- GRN after mixer output for inter-channel feature competition.
- Dual gating: SiLU (first gate) + Sigmoid (second gate).
- CLS-row architecture: CLS + 13 registers as extra row -> 15x14 grid.
- No RoPE — isolates GRN contribution without positional encoding in the gate.
"""

from examples.vit5_imagenet.v3_wessels._base_config import HIDDEN_DIM, build_cls_row_network, build_film_cfg, build_hyena_mixer, get_base_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.grn import GlobalResponseNorm


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena CLS-row + FiLM + GRN (no RoPE) config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_hyena_mixer(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(
        mixer_cfg,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=HIDDEN_DIM),
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
