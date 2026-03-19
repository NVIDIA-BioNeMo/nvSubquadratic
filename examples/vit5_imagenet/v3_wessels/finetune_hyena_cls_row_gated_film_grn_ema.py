"""ViT-5-Small + Hyena CLS-row + FiLM + GRN — ImageNet-1k fine-tuning.

Fine-tunes from pretrained run nxm3i7g6 (val/acc_ema 0.8173, 800 epochs).

Architecture: Depthwise Hyena with FiLM-conditioned SIREN kernels and
Global Response Normalization (GRN) after mixer output. CLS-row layout
with 13 registers. No RoPE.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    HIDDEN_DIM,
    build_cls_row_network,
    build_film_cfg,
    build_hyena_mixer,
)
from examples.vit5_imagenet.v3_wessels._finetune_base import get_finetune_base_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.grn import GlobalResponseNorm

PRETRAINED_RUN_ID = "nxm3i7g6"


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena CLS-row + FiLM + GRN finetune config."""
    config = get_finetune_base_config(pretrained_run_id=PRETRAINED_RUN_ID)

    film_cfg = build_film_cfg()
    mixer_cfg = build_hyena_mixer(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(
        mixer_cfg,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=HIDDEN_DIM),
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
