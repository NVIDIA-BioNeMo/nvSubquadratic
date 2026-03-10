"""ViT-5-Small + Hyena ImageNet-1k — GAP over registers, FiLM-conditioned SIREN.

Like the patch-GAP variant but classification pools over the 14 register tokens
instead of the 196 patch tokens. Ablation to compare register vs patch representations.

Token layout: [reg×14, patch×196] = 210 = 15×14 grid (same as patch-GAP).
Head: mean([B, 14, C], dim=1) → Linear → logits.
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
from nvsubquadratic.modules.register_reduction_head import RegisterGAPHead


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena register-GAP + FiLM config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_hyena_mixer(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(
        mixer_cfg,
        use_cls_token=False,
        num_registers=NUM_REGISTERS_NO_CLS,
        register_start_idx=0,
        register_head_cfg=LazyConfig(RegisterGAPHead)(
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
        ),
        find_unused_parameters=True,
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
