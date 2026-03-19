"""ViT-5-Small + Hyena register-head + FiLM — ImageNet-1k fine-tuning.

Fine-tunes from pretrained run oml9thv5 (val/acc_ema 0.8131, 800 epochs).

Architecture: Depthwise Hyena with FiLM-conditioned SIREN kernels and
register reduction head (Mamba-R style). Token layout: [reg×14, patch×196]
= 210 tokens = 15×14 grid. No CLS token, no RoPE, no GRN.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_REGISTERS_NO_CLS,
    build_cls_row_network,
    build_film_cfg,
    build_hyena_mixer,
)
from examples.vit5_imagenet.v3_wessels._finetune_base import get_finetune_base_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.register_reduction_head import RegisterReductionHead

PRETRAINED_RUN_ID = "oml9thv5"
REDUCTION_FACTOR = 4


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena register-head + FiLM finetune config."""
    config = get_finetune_base_config(pretrained_run_id=PRETRAINED_RUN_ID)

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
