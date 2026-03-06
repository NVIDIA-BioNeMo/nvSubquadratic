"""WSD finetuning ablation — no Mixup, no label smoothing.

Default smoothing is 0.1. With Mixup/CutMix already removed, label smoothing
may be unnecessary or even harmful for finetuning.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, no label smoothing."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0, smoothing=0.0)
