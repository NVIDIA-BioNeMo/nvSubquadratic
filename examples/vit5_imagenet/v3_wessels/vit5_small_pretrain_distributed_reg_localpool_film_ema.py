"""ViT-5-Small + Hyena ImageNet-1k — distributed registers + local pooling + FiLM.

Evenly distributes register tokens among patches (Mamba-R, arXiv:2405.14858).
Registers are stripped before Hyena's 2D convolution and updated via local
average pooling (each register reads from its 14 neighboring patches).

- CKConvND (depthwise) on pure 14x14 patch grid.
- FiLM-conditioned SIREN kernels (input-dependent via register pooling).
- Register local pooling: each register averages its stride=14 nearest patches.
- Register reduction head: Linear(384->96) per register, concat, Linear(1344->1000).
- Token layout: [P×14, R, P×14, R, ...] = 210 tokens (14 registers interleaved).
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_PATCHES,
    NUM_REGISTERS_NO_CLS,
    build_depthwise_hyena_mixer_patches_only,
    build_distributed_reg_network,
    build_film_cfg,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.register_patch_comm import RegisterLocalPooling
from nvsubquadratic.modules.register_reduction_head import RegisterReductionHead

REDUCTION_FACTOR = 4


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + distributed registers + local pooling + FiLM config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_depthwise_hyena_mixer_patches_only(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_distributed_reg_network(
        mixer_cfg,
        num_registers=NUM_REGISTERS_NO_CLS,
        register_head_cfg=LazyConfig(RegisterReductionHead)(
            hidden_dim=HIDDEN_DIM,
            num_registers=NUM_REGISTERS_NO_CLS,
            reduction_factor=REDUCTION_FACTOR,
            num_classes=NUM_CLASSES,
        ),
        register_comm_cfg=LazyConfig(RegisterLocalPooling)(
            hidden_dim=HIDDEN_DIM,
            num_registers=NUM_REGISTERS_NO_CLS,
            num_patches=NUM_PATCHES,
            use_proj=False,
        ),
        find_unused_parameters=True,
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
