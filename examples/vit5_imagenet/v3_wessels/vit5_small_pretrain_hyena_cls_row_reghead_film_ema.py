"""ViT-5-Small + Hyena ImageNet-1k — register reduction head + FiLM.

Combines register recycling (Mamba-R, arXiv:2405.14858) with FiLM conditioning:
- CKConvND (depthwise) with global channel mixing.
- FiLM-conditioned SIREN kernels (input-dependent via register pooling).
- Register reduction head: Linear(384->96) per register, concat, Linear(1344->1000).
- Token layout: [reg x 14, patch x 196] = 210 tokens = 15 x 14 grid.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_REGISTERS_NO_CLS,
    build_cls_row_network,
    build_film_cfg,
    build_hyena_mixer,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.register_reduction_head import RegisterReductionHead

REDUCTION_FACTOR = 4


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena register-recycling + FiLM config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_hyena_mixer(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(
        mixer_cfg,
        num_registers=NUM_REGISTERS_NO_CLS,
        use_cls_token=False,
        register_start_idx=0,
        register_head_cfg=LazyConfig(RegisterReductionHead)(
            hidden_dim=HIDDEN_DIM,
            num_registers=NUM_REGISTERS_NO_CLS,
            reduction_factor=REDUCTION_FACTOR,
            num_classes=NUM_CLASSES,
        ),
        find_unused_parameters=True,
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config