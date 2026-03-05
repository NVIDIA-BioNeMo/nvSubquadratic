"""WSD finetuning ablation — no Mixup, LR=3e-4, 5 epochs.

nomix_lr3e4 showed 81.96% EMA at epoch 2 — test if short training at high LR
captures the peak before degradation.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, LR=3e-4, 5 epochs."""
    return _base(lr=3e-4, wd=0.05, mixup=0.0, cutmix=0.0, epochs=5)
