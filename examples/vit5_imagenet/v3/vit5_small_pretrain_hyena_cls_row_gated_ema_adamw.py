"""ViT-5-Small + Hyena pretrain with AdamW optimizer (ablation).

Identical to the LAMB Hyena config but replaces FusedLAMB with AdamW.
Standard ViT hyperparameters: lr=1e-3, betas=(0.9, 0.999), wd=0.05, warmup=20ep.
Tests H3: whether LAMB's layerwise trust ratios mishandle SIREN parameters.
"""

import torch

from examples.vit5_imagenet.v3._pretrain_base import EPOCHS, PLACEHOLDER
from examples.vit5_imagenet.v3.vit5_small_pretrain_hyena_cls_row_gated_ema import (
    get_config as _get_hyena_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


ADAMW_LR = 1e-3
ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 0.05
ADAMW_WARMUP_EPOCHS = 20


def get_config() -> ExperimentConfig:
    """Return Hyena pretrain config with AdamW instead of LAMB."""
    config = _get_hyena_config()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        weight_decay=ADAMW_WEIGHT_DECAY,
    )

    config.scheduler.warmup_iterations_percentage = ADAMW_WARMUP_EPOCHS / EPOCHS

    return config
