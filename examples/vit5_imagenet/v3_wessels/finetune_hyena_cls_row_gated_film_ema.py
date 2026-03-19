"""ViT-5-Small + Hyena CLS-row + FiLM — ImageNet-1k fine-tuning.

Fine-tunes from pretrained run 69gwd0xm (val/acc_ema 0.8168, 800 epochs).

Architecture: Depthwise Hyena with FiLM-conditioned SIREN kernels.
CLS-row layout with 13 registers. No RoPE, no GRN.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    build_cls_row_network,
    build_film_cfg,
    build_hyena_mixer,
)
from examples.vit5_imagenet.v3_wessels._finetune_base import get_finetune_base_config
from experiments.default_cfg import ExperimentConfig

PRETRAINED_RUN_ID = "69gwd0xm"


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena CLS-row + FiLM finetune config."""
    config = get_finetune_base_config(pretrained_run_id=PRETRAINED_RUN_ID)

    film_cfg = build_film_cfg()
    mixer_cfg = build_hyena_mixer(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(mixer_cfg)
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
